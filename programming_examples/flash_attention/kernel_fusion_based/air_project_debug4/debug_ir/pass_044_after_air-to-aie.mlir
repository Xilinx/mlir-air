#map = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 65536)>
#map2 = affine_map<()[s0] -> (s0 * 65536 + 8192)>
#map3 = affine_map<()[s0] -> (s0 * 65536 + 16384)>
#map4 = affine_map<()[s0] -> (s0 * 65536 + 24576)>
#map5 = affine_map<()[s0, s1] -> (s0 * 65536 + s1 * 16384 + 32768)>
#map6 = affine_map<()[s0] -> (s0 * 65536 + 32768)>
#map7 = affine_map<()[s0] -> (s0 * 65536 + 40960)>
#map8 = affine_map<()[s0] -> (s0 * 65536 + 49152)>
#map9 = affine_map<()[s0] -> (s0 * 65536 + 57344)>
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
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 2 : i32}
    %lock_0_2_20 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_21 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_22 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_23 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_24 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 2 : i32}
    %lock_1_2_25 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_26 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_27 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_28 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_29 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 2 : i32}
    %lock_2_2_30 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_31 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_32 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_33 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_34 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 2 : i32}
    %lock_3_2_35 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_36 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_37 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_38 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_39 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 2 : i32}
    %lock_0_3_40 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_41 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_42 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 3) {init = 2 : i32}
    %lock_1_3_43 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_44 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_45 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 3) {init = 2 : i32}
    %lock_2_3_46 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_47 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_48 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 3) {init = 2 : i32}
    %lock_3_3_49 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_50 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_51 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 2 : i32}
    %lock_0_4_52 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_53 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_54 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 3) {init = 2 : i32}
    %lock_1_4_55 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_56 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_57 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 3) {init = 2 : i32}
    %lock_2_4_58 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_59 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_60 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 3) {init = 2 : i32}
    %lock_3_4_61 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_62 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_63 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 2 : i32}
    %lock_0_5_64 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_65 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_66 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 3) {init = 2 : i32}
    %lock_1_5_67 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_68 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_69 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 3) {init = 2 : i32}
    %lock_2_5_70 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_71 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_72 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 3) {init = 2 : i32}
    %lock_3_5_73 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_74 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_75 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf303 = aie.buffer(%mem_tile_0_1) {sym_name = "buf303"} : memref<64x64xbf16, 1 : i32> 
    %buf302 = aie.buffer(%mem_tile_1_1) {sym_name = "buf302"} : memref<64x64xbf16, 1 : i32> 
    %buf301 = aie.buffer(%mem_tile_2_1) {sym_name = "buf301"} : memref<64x64xbf16, 1 : i32> 
    %buf300 = aie.buffer(%mem_tile_3_1) {sym_name = "buf300"} : memref<64x64xbf16, 1 : i32> 
    %buf299 = aie.buffer(%mem_tile_0_1) {sym_name = "buf299"} : memref<64x64xbf16, 1 : i32> 
    %buf298 = aie.buffer(%mem_tile_0_1) {sym_name = "buf298"} : memref<64x64xbf16, 1 : i32> 
    %buf297 = aie.buffer(%mem_tile_1_1) {sym_name = "buf297"} : memref<64x64xbf16, 1 : i32> 
    %buf296 = aie.buffer(%mem_tile_1_1) {sym_name = "buf296"} : memref<64x64xbf16, 1 : i32> 
    %buf295 = aie.buffer(%mem_tile_2_1) {sym_name = "buf295"} : memref<64x64xbf16, 1 : i32> 
    %buf294 = aie.buffer(%mem_tile_2_1) {sym_name = "buf294"} : memref<64x64xbf16, 1 : i32> 
    %buf293 = aie.buffer(%mem_tile_3_1) {sym_name = "buf293"} : memref<64x64xbf16, 1 : i32> 
    %buf292 = aie.buffer(%mem_tile_3_1) {sym_name = "buf292"} : memref<64x64xbf16, 1 : i32> 
    %buf291 = aie.buffer(%tile_3_5) {sym_name = "buf291"} : memref<64x1xbf16, 2 : i32> 
    %buf290 = aie.buffer(%tile_3_5) {sym_name = "buf290"} : memref<64x1xbf16, 2 : i32> 
    %buf289 = aie.buffer(%tile_3_5) {sym_name = "buf289"} : memref<64x64xbf16, 2 : i32> 
    %buf288 = aie.buffer(%tile_3_5) {sym_name = "buf288"} : memref<64x64xbf16, 2 : i32> 
    %buf287 = aie.buffer(%tile_3_5) {sym_name = "buf287"} : memref<64x64xbf16, 2 : i32> 
    %buf286 = aie.buffer(%tile_3_5) {sym_name = "buf286"} : memref<64x64xbf16, 2 : i32> 
    %buf285 = aie.buffer(%tile_3_5) {sym_name = "buf285"} : memref<64x64xbf16, 2 : i32> 
    %buf284 = aie.buffer(%tile_3_5) {sym_name = "buf284"} : memref<64x1xbf16, 2 : i32> 
    %buf283 = aie.buffer(%tile_3_5) {sym_name = "buf283"} : memref<64x1xbf16, 2 : i32> 
    %buf282 = aie.buffer(%tile_3_5) {sym_name = "buf282"} : memref<64x64xbf16, 2 : i32> 
    %buf281 = aie.buffer(%tile_3_5) {sym_name = "buf281"} : memref<64x64xbf16, 2 : i32> 
    %buf280 = aie.buffer(%tile_3_5) {sym_name = "buf280"} : memref<64x1xbf16, 2 : i32> 
    %buf279 = aie.buffer(%tile_3_5) {sym_name = "buf279"} : memref<64x1xbf16, 2 : i32> 
    %buf278 = aie.buffer(%tile_2_5) {sym_name = "buf278"} : memref<64x1xbf16, 2 : i32> 
    %buf277 = aie.buffer(%tile_2_5) {sym_name = "buf277"} : memref<64x1xbf16, 2 : i32> 
    %buf276 = aie.buffer(%tile_2_5) {sym_name = "buf276"} : memref<64x64xbf16, 2 : i32> 
    %buf275 = aie.buffer(%tile_2_5) {sym_name = "buf275"} : memref<64x64xbf16, 2 : i32> 
    %buf274 = aie.buffer(%tile_2_5) {sym_name = "buf274"} : memref<64x64xbf16, 2 : i32> 
    %buf273 = aie.buffer(%tile_2_5) {sym_name = "buf273"} : memref<64x64xbf16, 2 : i32> 
    %buf272 = aie.buffer(%tile_2_5) {sym_name = "buf272"} : memref<64x64xbf16, 2 : i32> 
    %buf271 = aie.buffer(%tile_2_5) {sym_name = "buf271"} : memref<64x1xbf16, 2 : i32> 
    %buf270 = aie.buffer(%tile_2_5) {sym_name = "buf270"} : memref<64x1xbf16, 2 : i32> 
    %buf269 = aie.buffer(%tile_2_5) {sym_name = "buf269"} : memref<64x64xbf16, 2 : i32> 
    %buf268 = aie.buffer(%tile_2_5) {sym_name = "buf268"} : memref<64x64xbf16, 2 : i32> 
    %buf267 = aie.buffer(%tile_2_5) {sym_name = "buf267"} : memref<64x1xbf16, 2 : i32> 
    %buf266 = aie.buffer(%tile_2_5) {sym_name = "buf266"} : memref<64x1xbf16, 2 : i32> 
    %buf265 = aie.buffer(%tile_1_5) {sym_name = "buf265"} : memref<64x1xbf16, 2 : i32> 
    %buf264 = aie.buffer(%tile_1_5) {sym_name = "buf264"} : memref<64x1xbf16, 2 : i32> 
    %buf263 = aie.buffer(%tile_1_5) {sym_name = "buf263"} : memref<64x64xbf16, 2 : i32> 
    %buf262 = aie.buffer(%tile_1_5) {sym_name = "buf262"} : memref<64x64xbf16, 2 : i32> 
    %buf261 = aie.buffer(%tile_1_5) {sym_name = "buf261"} : memref<64x64xbf16, 2 : i32> 
    %buf260 = aie.buffer(%tile_1_5) {sym_name = "buf260"} : memref<64x64xbf16, 2 : i32> 
    %buf259 = aie.buffer(%tile_1_5) {sym_name = "buf259"} : memref<64x64xbf16, 2 : i32> 
    %buf258 = aie.buffer(%tile_1_5) {sym_name = "buf258"} : memref<64x1xbf16, 2 : i32> 
    %buf257 = aie.buffer(%tile_1_5) {sym_name = "buf257"} : memref<64x1xbf16, 2 : i32> 
    %buf256 = aie.buffer(%tile_1_5) {sym_name = "buf256"} : memref<64x64xbf16, 2 : i32> 
    %buf255 = aie.buffer(%tile_1_5) {sym_name = "buf255"} : memref<64x64xbf16, 2 : i32> 
    %buf254 = aie.buffer(%tile_1_5) {sym_name = "buf254"} : memref<64x1xbf16, 2 : i32> 
    %buf253 = aie.buffer(%tile_1_5) {sym_name = "buf253"} : memref<64x1xbf16, 2 : i32> 
    %buf252 = aie.buffer(%tile_0_5) {sym_name = "buf252"} : memref<64x1xbf16, 2 : i32> 
    %buf251 = aie.buffer(%tile_0_5) {sym_name = "buf251"} : memref<64x1xbf16, 2 : i32> 
    %buf250 = aie.buffer(%tile_0_5) {sym_name = "buf250"} : memref<64x64xbf16, 2 : i32> 
    %buf249 = aie.buffer(%tile_0_5) {sym_name = "buf249"} : memref<64x64xbf16, 2 : i32> 
    %buf248 = aie.buffer(%tile_0_5) {sym_name = "buf248"} : memref<64x64xbf16, 2 : i32> 
    %buf247 = aie.buffer(%tile_0_5) {sym_name = "buf247"} : memref<64x64xbf16, 2 : i32> 
    %buf246 = aie.buffer(%tile_0_5) {sym_name = "buf246"} : memref<64x64xbf16, 2 : i32> 
    %buf245 = aie.buffer(%tile_0_5) {sym_name = "buf245"} : memref<64x1xbf16, 2 : i32> 
    %buf244 = aie.buffer(%tile_0_5) {sym_name = "buf244"} : memref<64x1xbf16, 2 : i32> 
    %buf243 = aie.buffer(%tile_0_5) {sym_name = "buf243"} : memref<64x64xbf16, 2 : i32> 
    %buf242 = aie.buffer(%tile_0_5) {sym_name = "buf242"} : memref<64x64xbf16, 2 : i32> 
    %buf241 = aie.buffer(%tile_0_5) {sym_name = "buf241"} : memref<64x1xbf16, 2 : i32> 
    %buf240 = aie.buffer(%tile_0_5) {sym_name = "buf240"} : memref<64x1xbf16, 2 : i32> 
    %buf239 = aie.buffer(%tile_3_4) {sym_name = "buf239"} : memref<64x1xbf16, 2 : i32> 
    %buf238 = aie.buffer(%tile_3_4) {sym_name = "buf238"} : memref<64x1xbf16, 2 : i32> 
    %buf237 = aie.buffer(%tile_3_4) {sym_name = "buf237"} : memref<64x64xbf16, 2 : i32> 
    %buf236 = aie.buffer(%tile_3_4) {sym_name = "buf236"} : memref<64x64xbf16, 2 : i32> 
    %buf235 = aie.buffer(%tile_3_4) {sym_name = "buf235"} : memref<64x64xbf16, 2 : i32> 
    %buf234 = aie.buffer(%tile_3_4) {sym_name = "buf234"} : memref<64x64xbf16, 2 : i32> 
    %buf233 = aie.buffer(%tile_3_4) {sym_name = "buf233"} : memref<64x64xbf16, 2 : i32> 
    %buf232 = aie.buffer(%tile_3_4) {sym_name = "buf232"} : memref<64x1xbf16, 2 : i32> 
    %buf231 = aie.buffer(%tile_3_4) {sym_name = "buf231"} : memref<64x1xbf16, 2 : i32> 
    %buf230 = aie.buffer(%tile_3_4) {sym_name = "buf230"} : memref<64x64xbf16, 2 : i32> 
    %buf229 = aie.buffer(%tile_3_4) {sym_name = "buf229"} : memref<64x64xbf16, 2 : i32> 
    %buf228 = aie.buffer(%tile_3_4) {sym_name = "buf228"} : memref<64x1xbf16, 2 : i32> 
    %buf227 = aie.buffer(%tile_3_4) {sym_name = "buf227"} : memref<64x1xbf16, 2 : i32> 
    %buf226 = aie.buffer(%tile_3_4) {sym_name = "buf226"} : memref<64x64xbf16, 2 : i32> 
    %buf225 = aie.buffer(%tile_3_4) {sym_name = "buf225"} : memref<64x1xbf16, 2 : i32> 
    %buf224 = aie.buffer(%tile_3_4) {sym_name = "buf224"} : memref<64x1xbf16, 2 : i32> 
    %buf223 = aie.buffer(%tile_3_4) {sym_name = "buf223"} : memref<64x1xbf16, 2 : i32> 
    %buf222 = aie.buffer(%tile_3_4) {sym_name = "buf222"} : memref<64x1xbf16, 2 : i32> 
    %buf221 = aie.buffer(%tile_3_4) {sym_name = "buf221"} : memref<64x1xbf16, 2 : i32> 
    %buf220 = aie.buffer(%tile_3_4) {sym_name = "buf220"} : memref<64x1xbf16, 2 : i32> 
    %buf219 = aie.buffer(%tile_2_4) {sym_name = "buf219"} : memref<64x1xbf16, 2 : i32> 
    %buf218 = aie.buffer(%tile_2_4) {sym_name = "buf218"} : memref<64x1xbf16, 2 : i32> 
    %buf217 = aie.buffer(%tile_2_4) {sym_name = "buf217"} : memref<64x64xbf16, 2 : i32> 
    %buf216 = aie.buffer(%tile_2_4) {sym_name = "buf216"} : memref<64x64xbf16, 2 : i32> 
    %buf215 = aie.buffer(%tile_2_4) {sym_name = "buf215"} : memref<64x64xbf16, 2 : i32> 
    %buf214 = aie.buffer(%tile_2_4) {sym_name = "buf214"} : memref<64x64xbf16, 2 : i32> 
    %buf213 = aie.buffer(%tile_2_4) {sym_name = "buf213"} : memref<64x64xbf16, 2 : i32> 
    %buf212 = aie.buffer(%tile_2_4) {sym_name = "buf212"} : memref<64x1xbf16, 2 : i32> 
    %buf211 = aie.buffer(%tile_2_4) {sym_name = "buf211"} : memref<64x1xbf16, 2 : i32> 
    %buf210 = aie.buffer(%tile_2_4) {sym_name = "buf210"} : memref<64x64xbf16, 2 : i32> 
    %buf209 = aie.buffer(%tile_2_4) {sym_name = "buf209"} : memref<64x64xbf16, 2 : i32> 
    %buf208 = aie.buffer(%tile_2_4) {sym_name = "buf208"} : memref<64x1xbf16, 2 : i32> 
    %buf207 = aie.buffer(%tile_2_4) {sym_name = "buf207"} : memref<64x1xbf16, 2 : i32> 
    %buf206 = aie.buffer(%tile_2_4) {sym_name = "buf206"} : memref<64x64xbf16, 2 : i32> 
    %buf205 = aie.buffer(%tile_2_4) {sym_name = "buf205"} : memref<64x1xbf16, 2 : i32> 
    %buf204 = aie.buffer(%tile_2_4) {sym_name = "buf204"} : memref<64x1xbf16, 2 : i32> 
    %buf203 = aie.buffer(%tile_2_4) {sym_name = "buf203"} : memref<64x1xbf16, 2 : i32> 
    %buf202 = aie.buffer(%tile_2_4) {sym_name = "buf202"} : memref<64x1xbf16, 2 : i32> 
    %buf201 = aie.buffer(%tile_2_4) {sym_name = "buf201"} : memref<64x1xbf16, 2 : i32> 
    %buf200 = aie.buffer(%tile_2_4) {sym_name = "buf200"} : memref<64x1xbf16, 2 : i32> 
    %buf199 = aie.buffer(%tile_1_4) {sym_name = "buf199"} : memref<64x1xbf16, 2 : i32> 
    %buf198 = aie.buffer(%tile_1_4) {sym_name = "buf198"} : memref<64x1xbf16, 2 : i32> 
    %buf197 = aie.buffer(%tile_1_4) {sym_name = "buf197"} : memref<64x64xbf16, 2 : i32> 
    %buf196 = aie.buffer(%tile_1_4) {sym_name = "buf196"} : memref<64x64xbf16, 2 : i32> 
    %buf195 = aie.buffer(%tile_1_4) {sym_name = "buf195"} : memref<64x64xbf16, 2 : i32> 
    %buf194 = aie.buffer(%tile_1_4) {sym_name = "buf194"} : memref<64x64xbf16, 2 : i32> 
    %buf193 = aie.buffer(%tile_1_4) {sym_name = "buf193"} : memref<64x64xbf16, 2 : i32> 
    %buf192 = aie.buffer(%tile_1_4) {sym_name = "buf192"} : memref<64x1xbf16, 2 : i32> 
    %buf191 = aie.buffer(%tile_1_4) {sym_name = "buf191"} : memref<64x1xbf16, 2 : i32> 
    %buf190 = aie.buffer(%tile_1_4) {sym_name = "buf190"} : memref<64x64xbf16, 2 : i32> 
    %buf189 = aie.buffer(%tile_1_4) {sym_name = "buf189"} : memref<64x64xbf16, 2 : i32> 
    %buf188 = aie.buffer(%tile_1_4) {sym_name = "buf188"} : memref<64x1xbf16, 2 : i32> 
    %buf187 = aie.buffer(%tile_1_4) {sym_name = "buf187"} : memref<64x1xbf16, 2 : i32> 
    %buf186 = aie.buffer(%tile_1_4) {sym_name = "buf186"} : memref<64x64xbf16, 2 : i32> 
    %buf185 = aie.buffer(%tile_1_4) {sym_name = "buf185"} : memref<64x1xbf16, 2 : i32> 
    %buf184 = aie.buffer(%tile_1_4) {sym_name = "buf184"} : memref<64x1xbf16, 2 : i32> 
    %buf183 = aie.buffer(%tile_1_4) {sym_name = "buf183"} : memref<64x1xbf16, 2 : i32> 
    %buf182 = aie.buffer(%tile_1_4) {sym_name = "buf182"} : memref<64x1xbf16, 2 : i32> 
    %buf181 = aie.buffer(%tile_1_4) {sym_name = "buf181"} : memref<64x1xbf16, 2 : i32> 
    %buf180 = aie.buffer(%tile_1_4) {sym_name = "buf180"} : memref<64x1xbf16, 2 : i32> 
    %buf179 = aie.buffer(%tile_0_4) {sym_name = "buf179"} : memref<64x1xbf16, 2 : i32> 
    %buf178 = aie.buffer(%tile_0_4) {sym_name = "buf178"} : memref<64x1xbf16, 2 : i32> 
    %buf177 = aie.buffer(%tile_0_4) {sym_name = "buf177"} : memref<64x64xbf16, 2 : i32> 
    %buf176 = aie.buffer(%tile_0_4) {sym_name = "buf176"} : memref<64x64xbf16, 2 : i32> 
    %buf175 = aie.buffer(%tile_0_4) {sym_name = "buf175"} : memref<64x64xbf16, 2 : i32> 
    %buf174 = aie.buffer(%tile_0_4) {sym_name = "buf174"} : memref<64x64xbf16, 2 : i32> 
    %buf173 = aie.buffer(%tile_0_4) {sym_name = "buf173"} : memref<64x64xbf16, 2 : i32> 
    %buf172 = aie.buffer(%tile_0_4) {sym_name = "buf172"} : memref<64x1xbf16, 2 : i32> 
    %buf171 = aie.buffer(%tile_0_4) {sym_name = "buf171"} : memref<64x1xbf16, 2 : i32> 
    %buf170 = aie.buffer(%tile_0_4) {sym_name = "buf170"} : memref<64x64xbf16, 2 : i32> 
    %buf169 = aie.buffer(%tile_0_4) {sym_name = "buf169"} : memref<64x64xbf16, 2 : i32> 
    %buf168 = aie.buffer(%tile_0_4) {sym_name = "buf168"} : memref<64x1xbf16, 2 : i32> 
    %buf167 = aie.buffer(%tile_0_4) {sym_name = "buf167"} : memref<64x1xbf16, 2 : i32> 
    %buf166 = aie.buffer(%tile_0_4) {sym_name = "buf166"} : memref<64x64xbf16, 2 : i32> 
    %buf165 = aie.buffer(%tile_0_4) {sym_name = "buf165"} : memref<64x1xbf16, 2 : i32> 
    %buf164 = aie.buffer(%tile_0_4) {sym_name = "buf164"} : memref<64x1xbf16, 2 : i32> 
    %buf163 = aie.buffer(%tile_0_4) {sym_name = "buf163"} : memref<64x1xbf16, 2 : i32> 
    %buf162 = aie.buffer(%tile_0_4) {sym_name = "buf162"} : memref<64x1xbf16, 2 : i32> 
    %buf161 = aie.buffer(%tile_0_4) {sym_name = "buf161"} : memref<64x1xbf16, 2 : i32> 
    %buf160 = aie.buffer(%tile_0_4) {sym_name = "buf160"} : memref<64x1xbf16, 2 : i32> 
    %buf159 = aie.buffer(%tile_3_3) {sym_name = "buf159"} : memref<64x1xbf16, 2 : i32> 
    %buf158 = aie.buffer(%tile_3_3) {sym_name = "buf158"} : memref<64x1xbf16, 2 : i32> 
    %buf157 = aie.buffer(%tile_3_3) {sym_name = "buf157"} : memref<64x64xbf16, 2 : i32> 
    %buf156 = aie.buffer(%tile_3_3) {sym_name = "buf156"} : memref<64x64xbf16, 2 : i32> 
    %buf155 = aie.buffer(%tile_3_3) {sym_name = "buf155"} : memref<64x64xbf16, 2 : i32> 
    %buf154 = aie.buffer(%tile_3_3) {sym_name = "buf154"} : memref<64x64xbf16, 2 : i32> 
    %buf153 = aie.buffer(%tile_3_3) {sym_name = "buf153"} : memref<64x64xbf16, 2 : i32> 
    %buf152 = aie.buffer(%tile_3_3) {sym_name = "buf152"} : memref<64x1xbf16, 2 : i32> 
    %buf151 = aie.buffer(%tile_3_3) {sym_name = "buf151"} : memref<64x1xbf16, 2 : i32> 
    %buf150 = aie.buffer(%tile_3_3) {sym_name = "buf150"} : memref<64x64xbf16, 2 : i32> 
    %buf149 = aie.buffer(%tile_3_3) {sym_name = "buf149"} : memref<64x64xbf16, 2 : i32> 
    %buf148 = aie.buffer(%tile_3_3) {sym_name = "buf148"} : memref<64x1xbf16, 2 : i32> 
    %buf147 = aie.buffer(%tile_3_3) {sym_name = "buf147"} : memref<64x1xbf16, 2 : i32> 
    %buf146 = aie.buffer(%tile_3_3) {sym_name = "buf146"} : memref<64x64xbf16, 2 : i32> 
    %buf145 = aie.buffer(%tile_3_3) {sym_name = "buf145"} : memref<64x1xbf16, 2 : i32> 
    %buf144 = aie.buffer(%tile_3_3) {sym_name = "buf144"} : memref<64x1xbf16, 2 : i32> 
    %buf143 = aie.buffer(%tile_3_3) {sym_name = "buf143"} : memref<64x1xbf16, 2 : i32> 
    %buf142 = aie.buffer(%tile_3_3) {sym_name = "buf142"} : memref<64x1xbf16, 2 : i32> 
    %buf141 = aie.buffer(%tile_3_3) {sym_name = "buf141"} : memref<64x1xbf16, 2 : i32> 
    %buf140 = aie.buffer(%tile_3_3) {sym_name = "buf140"} : memref<64x1xbf16, 2 : i32> 
    %buf139 = aie.buffer(%tile_2_3) {sym_name = "buf139"} : memref<64x1xbf16, 2 : i32> 
    %buf138 = aie.buffer(%tile_2_3) {sym_name = "buf138"} : memref<64x1xbf16, 2 : i32> 
    %buf137 = aie.buffer(%tile_2_3) {sym_name = "buf137"} : memref<64x64xbf16, 2 : i32> 
    %buf136 = aie.buffer(%tile_2_3) {sym_name = "buf136"} : memref<64x64xbf16, 2 : i32> 
    %buf135 = aie.buffer(%tile_2_3) {sym_name = "buf135"} : memref<64x64xbf16, 2 : i32> 
    %buf134 = aie.buffer(%tile_2_3) {sym_name = "buf134"} : memref<64x64xbf16, 2 : i32> 
    %buf133 = aie.buffer(%tile_2_3) {sym_name = "buf133"} : memref<64x64xbf16, 2 : i32> 
    %buf132 = aie.buffer(%tile_2_3) {sym_name = "buf132"} : memref<64x1xbf16, 2 : i32> 
    %buf131 = aie.buffer(%tile_2_3) {sym_name = "buf131"} : memref<64x1xbf16, 2 : i32> 
    %buf130 = aie.buffer(%tile_2_3) {sym_name = "buf130"} : memref<64x64xbf16, 2 : i32> 
    %buf129 = aie.buffer(%tile_2_3) {sym_name = "buf129"} : memref<64x64xbf16, 2 : i32> 
    %buf128 = aie.buffer(%tile_2_3) {sym_name = "buf128"} : memref<64x1xbf16, 2 : i32> 
    %buf127 = aie.buffer(%tile_2_3) {sym_name = "buf127"} : memref<64x1xbf16, 2 : i32> 
    %buf126 = aie.buffer(%tile_2_3) {sym_name = "buf126"} : memref<64x64xbf16, 2 : i32> 
    %buf125 = aie.buffer(%tile_2_3) {sym_name = "buf125"} : memref<64x1xbf16, 2 : i32> 
    %buf124 = aie.buffer(%tile_2_3) {sym_name = "buf124"} : memref<64x1xbf16, 2 : i32> 
    %buf123 = aie.buffer(%tile_2_3) {sym_name = "buf123"} : memref<64x1xbf16, 2 : i32> 
    %buf122 = aie.buffer(%tile_2_3) {sym_name = "buf122"} : memref<64x1xbf16, 2 : i32> 
    %buf121 = aie.buffer(%tile_2_3) {sym_name = "buf121"} : memref<64x1xbf16, 2 : i32> 
    %buf120 = aie.buffer(%tile_2_3) {sym_name = "buf120"} : memref<64x1xbf16, 2 : i32> 
    %buf119 = aie.buffer(%tile_1_3) {sym_name = "buf119"} : memref<64x1xbf16, 2 : i32> 
    %buf118 = aie.buffer(%tile_1_3) {sym_name = "buf118"} : memref<64x1xbf16, 2 : i32> 
    %buf117 = aie.buffer(%tile_1_3) {sym_name = "buf117"} : memref<64x64xbf16, 2 : i32> 
    %buf116 = aie.buffer(%tile_1_3) {sym_name = "buf116"} : memref<64x64xbf16, 2 : i32> 
    %buf115 = aie.buffer(%tile_1_3) {sym_name = "buf115"} : memref<64x64xbf16, 2 : i32> 
    %buf114 = aie.buffer(%tile_1_3) {sym_name = "buf114"} : memref<64x64xbf16, 2 : i32> 
    %buf113 = aie.buffer(%tile_1_3) {sym_name = "buf113"} : memref<64x64xbf16, 2 : i32> 
    %buf112 = aie.buffer(%tile_1_3) {sym_name = "buf112"} : memref<64x1xbf16, 2 : i32> 
    %buf111 = aie.buffer(%tile_1_3) {sym_name = "buf111"} : memref<64x1xbf16, 2 : i32> 
    %buf110 = aie.buffer(%tile_1_3) {sym_name = "buf110"} : memref<64x64xbf16, 2 : i32> 
    %buf109 = aie.buffer(%tile_1_3) {sym_name = "buf109"} : memref<64x64xbf16, 2 : i32> 
    %buf108 = aie.buffer(%tile_1_3) {sym_name = "buf108"} : memref<64x1xbf16, 2 : i32> 
    %buf107 = aie.buffer(%tile_1_3) {sym_name = "buf107"} : memref<64x1xbf16, 2 : i32> 
    %buf106 = aie.buffer(%tile_1_3) {sym_name = "buf106"} : memref<64x64xbf16, 2 : i32> 
    %buf105 = aie.buffer(%tile_1_3) {sym_name = "buf105"} : memref<64x1xbf16, 2 : i32> 
    %buf104 = aie.buffer(%tile_1_3) {sym_name = "buf104"} : memref<64x1xbf16, 2 : i32> 
    %buf103 = aie.buffer(%tile_1_3) {sym_name = "buf103"} : memref<64x1xbf16, 2 : i32> 
    %buf102 = aie.buffer(%tile_1_3) {sym_name = "buf102"} : memref<64x1xbf16, 2 : i32> 
    %buf101 = aie.buffer(%tile_1_3) {sym_name = "buf101"} : memref<64x1xbf16, 2 : i32> 
    %buf100 = aie.buffer(%tile_1_3) {sym_name = "buf100"} : memref<64x1xbf16, 2 : i32> 
    %buf99 = aie.buffer(%tile_0_3) {sym_name = "buf99"} : memref<64x1xbf16, 2 : i32> 
    %buf98 = aie.buffer(%tile_0_3) {sym_name = "buf98"} : memref<64x1xbf16, 2 : i32> 
    %buf97 = aie.buffer(%tile_0_3) {sym_name = "buf97"} : memref<64x64xbf16, 2 : i32> 
    %buf96 = aie.buffer(%tile_0_3) {sym_name = "buf96"} : memref<64x64xbf16, 2 : i32> 
    %buf95 = aie.buffer(%tile_0_3) {sym_name = "buf95"} : memref<64x64xbf16, 2 : i32> 
    %buf94 = aie.buffer(%tile_0_3) {sym_name = "buf94"} : memref<64x64xbf16, 2 : i32> 
    %buf93 = aie.buffer(%tile_0_3) {sym_name = "buf93"} : memref<64x64xbf16, 2 : i32> 
    %buf92 = aie.buffer(%tile_0_3) {sym_name = "buf92"} : memref<64x1xbf16, 2 : i32> 
    %buf91 = aie.buffer(%tile_0_3) {sym_name = "buf91"} : memref<64x1xbf16, 2 : i32> 
    %buf90 = aie.buffer(%tile_0_3) {sym_name = "buf90"} : memref<64x64xbf16, 2 : i32> 
    %buf89 = aie.buffer(%tile_0_3) {sym_name = "buf89"} : memref<64x64xbf16, 2 : i32> 
    %buf88 = aie.buffer(%tile_0_3) {sym_name = "buf88"} : memref<64x1xbf16, 2 : i32> 
    %buf87 = aie.buffer(%tile_0_3) {sym_name = "buf87"} : memref<64x1xbf16, 2 : i32> 
    %buf86 = aie.buffer(%tile_0_3) {sym_name = "buf86"} : memref<64x64xbf16, 2 : i32> 
    %buf85 = aie.buffer(%tile_0_3) {sym_name = "buf85"} : memref<64x1xbf16, 2 : i32> 
    %buf84 = aie.buffer(%tile_0_3) {sym_name = "buf84"} : memref<64x1xbf16, 2 : i32> 
    %buf83 = aie.buffer(%tile_0_3) {sym_name = "buf83"} : memref<64x1xbf16, 2 : i32> 
    %buf82 = aie.buffer(%tile_0_3) {sym_name = "buf82"} : memref<64x1xbf16, 2 : i32> 
    %buf81 = aie.buffer(%tile_0_3) {sym_name = "buf81"} : memref<64x1xbf16, 2 : i32> 
    %buf80 = aie.buffer(%tile_0_3) {sym_name = "buf80"} : memref<64x1xbf16, 2 : i32> 
    %buf79 = aie.buffer(%tile_3_2) {sym_name = "buf79"} : memref<64x1xbf16, 2 : i32> 
    %buf78 = aie.buffer(%tile_3_2) {sym_name = "buf78"} : memref<64x1xbf16, 2 : i32> 
    %buf77 = aie.buffer(%tile_3_2) {sym_name = "buf77"} : memref<64x64xbf16, 2 : i32> 
    %buf76 = aie.buffer(%tile_3_2) {sym_name = "buf76"} : memref<64x64xbf16, 2 : i32> 
    %buf75 = aie.buffer(%tile_3_2) {sym_name = "buf75"} : memref<64x64xbf16, 2 : i32> 
    %buf74 = aie.buffer(%tile_3_2) {sym_name = "buf74"} : memref<64x64xbf16, 2 : i32> 
    %buf73 = aie.buffer(%tile_3_2) {sym_name = "buf73"} : memref<64x64xbf16, 2 : i32> 
    %buf72 = aie.buffer(%tile_3_2) {sym_name = "buf72"} : memref<64x1xbf16, 2 : i32> 
    %buf71 = aie.buffer(%tile_3_2) {sym_name = "buf71"} : memref<64x1xbf16, 2 : i32> 
    %buf70 = aie.buffer(%tile_3_2) {sym_name = "buf70"} : memref<64x64xbf16, 2 : i32> 
    %buf69 = aie.buffer(%tile_3_2) {sym_name = "buf69"} : memref<64x64xbf16, 2 : i32> 
    %buf68 = aie.buffer(%tile_3_2) {sym_name = "buf68"} : memref<64x1xbf16, 2 : i32> 
    %buf67 = aie.buffer(%tile_3_2) {sym_name = "buf67"} : memref<64x1xbf16, 2 : i32> 
    %buf66 = aie.buffer(%tile_3_2) {sym_name = "buf66"} : memref<64x64xbf16, 2 : i32> 
    %buf65 = aie.buffer(%tile_3_2) {sym_name = "buf65"} : memref<64x1xbf16, 2 : i32> 
    %buf64 = aie.buffer(%tile_3_2) {sym_name = "buf64"} : memref<64x1xbf16, 2 : i32> 
    %buf63 = aie.buffer(%tile_3_2) {sym_name = "buf63"} : memref<64x1xbf16, 2 : i32> 
    %buf62 = aie.buffer(%tile_3_2) {sym_name = "buf62"} : memref<64x1xbf16, 2 : i32> 
    %buf61 = aie.buffer(%tile_3_2) {sym_name = "buf61"} : memref<64x1xbf16, 2 : i32> 
    %buf60 = aie.buffer(%tile_3_2) {sym_name = "buf60"} : memref<64x1xbf16, 2 : i32> 
    %buf59 = aie.buffer(%tile_2_2) {sym_name = "buf59"} : memref<64x1xbf16, 2 : i32> 
    %buf58 = aie.buffer(%tile_2_2) {sym_name = "buf58"} : memref<64x1xbf16, 2 : i32> 
    %buf57 = aie.buffer(%tile_2_2) {sym_name = "buf57"} : memref<64x64xbf16, 2 : i32> 
    %buf56 = aie.buffer(%tile_2_2) {sym_name = "buf56"} : memref<64x64xbf16, 2 : i32> 
    %buf55 = aie.buffer(%tile_2_2) {sym_name = "buf55"} : memref<64x64xbf16, 2 : i32> 
    %buf54 = aie.buffer(%tile_2_2) {sym_name = "buf54"} : memref<64x64xbf16, 2 : i32> 
    %buf53 = aie.buffer(%tile_2_2) {sym_name = "buf53"} : memref<64x64xbf16, 2 : i32> 
    %buf52 = aie.buffer(%tile_2_2) {sym_name = "buf52"} : memref<64x1xbf16, 2 : i32> 
    %buf51 = aie.buffer(%tile_2_2) {sym_name = "buf51"} : memref<64x1xbf16, 2 : i32> 
    %buf50 = aie.buffer(%tile_2_2) {sym_name = "buf50"} : memref<64x64xbf16, 2 : i32> 
    %buf49 = aie.buffer(%tile_2_2) {sym_name = "buf49"} : memref<64x64xbf16, 2 : i32> 
    %buf48 = aie.buffer(%tile_2_2) {sym_name = "buf48"} : memref<64x1xbf16, 2 : i32> 
    %buf47 = aie.buffer(%tile_2_2) {sym_name = "buf47"} : memref<64x1xbf16, 2 : i32> 
    %buf46 = aie.buffer(%tile_2_2) {sym_name = "buf46"} : memref<64x64xbf16, 2 : i32> 
    %buf45 = aie.buffer(%tile_2_2) {sym_name = "buf45"} : memref<64x1xbf16, 2 : i32> 
    %buf44 = aie.buffer(%tile_2_2) {sym_name = "buf44"} : memref<64x1xbf16, 2 : i32> 
    %buf43 = aie.buffer(%tile_2_2) {sym_name = "buf43"} : memref<64x1xbf16, 2 : i32> 
    %buf42 = aie.buffer(%tile_2_2) {sym_name = "buf42"} : memref<64x1xbf16, 2 : i32> 
    %buf41 = aie.buffer(%tile_2_2) {sym_name = "buf41"} : memref<64x1xbf16, 2 : i32> 
    %buf40 = aie.buffer(%tile_2_2) {sym_name = "buf40"} : memref<64x1xbf16, 2 : i32> 
    %buf39 = aie.buffer(%tile_1_2) {sym_name = "buf39"} : memref<64x1xbf16, 2 : i32> 
    %buf38 = aie.buffer(%tile_1_2) {sym_name = "buf38"} : memref<64x1xbf16, 2 : i32> 
    %buf37 = aie.buffer(%tile_1_2) {sym_name = "buf37"} : memref<64x64xbf16, 2 : i32> 
    %buf36 = aie.buffer(%tile_1_2) {sym_name = "buf36"} : memref<64x64xbf16, 2 : i32> 
    %buf35 = aie.buffer(%tile_1_2) {sym_name = "buf35"} : memref<64x64xbf16, 2 : i32> 
    %buf34 = aie.buffer(%tile_1_2) {sym_name = "buf34"} : memref<64x64xbf16, 2 : i32> 
    %buf33 = aie.buffer(%tile_1_2) {sym_name = "buf33"} : memref<64x64xbf16, 2 : i32> 
    %buf32 = aie.buffer(%tile_1_2) {sym_name = "buf32"} : memref<64x1xbf16, 2 : i32> 
    %buf31 = aie.buffer(%tile_1_2) {sym_name = "buf31"} : memref<64x1xbf16, 2 : i32> 
    %buf30 = aie.buffer(%tile_1_2) {sym_name = "buf30"} : memref<64x64xbf16, 2 : i32> 
    %buf29 = aie.buffer(%tile_1_2) {sym_name = "buf29"} : memref<64x64xbf16, 2 : i32> 
    %buf28 = aie.buffer(%tile_1_2) {sym_name = "buf28"} : memref<64x1xbf16, 2 : i32> 
    %buf27 = aie.buffer(%tile_1_2) {sym_name = "buf27"} : memref<64x1xbf16, 2 : i32> 
    %buf26 = aie.buffer(%tile_1_2) {sym_name = "buf26"} : memref<64x64xbf16, 2 : i32> 
    %buf25 = aie.buffer(%tile_1_2) {sym_name = "buf25"} : memref<64x1xbf16, 2 : i32> 
    %buf24 = aie.buffer(%tile_1_2) {sym_name = "buf24"} : memref<64x1xbf16, 2 : i32> 
    %buf23 = aie.buffer(%tile_1_2) {sym_name = "buf23"} : memref<64x1xbf16, 2 : i32> 
    %buf22 = aie.buffer(%tile_1_2) {sym_name = "buf22"} : memref<64x1xbf16, 2 : i32> 
    %buf21 = aie.buffer(%tile_1_2) {sym_name = "buf21"} : memref<64x1xbf16, 2 : i32> 
    %buf20 = aie.buffer(%tile_1_2) {sym_name = "buf20"} : memref<64x1xbf16, 2 : i32> 
    %buf19 = aie.buffer(%tile_0_2) {sym_name = "buf19"} : memref<64x1xbf16, 2 : i32> 
    %buf18 = aie.buffer(%tile_0_2) {sym_name = "buf18"} : memref<64x1xbf16, 2 : i32> 
    %buf17 = aie.buffer(%tile_0_2) {sym_name = "buf17"} : memref<64x64xbf16, 2 : i32> 
    %buf16 = aie.buffer(%tile_0_2) {sym_name = "buf16"} : memref<64x64xbf16, 2 : i32> 
    %buf15 = aie.buffer(%tile_0_2) {sym_name = "buf15"} : memref<64x64xbf16, 2 : i32> 
    %buf14 = aie.buffer(%tile_0_2) {sym_name = "buf14"} : memref<64x64xbf16, 2 : i32> 
    %buf13 = aie.buffer(%tile_0_2) {sym_name = "buf13"} : memref<64x64xbf16, 2 : i32> 
    %buf12 = aie.buffer(%tile_0_2) {sym_name = "buf12"} : memref<64x1xbf16, 2 : i32> 
    %buf11 = aie.buffer(%tile_0_2) {sym_name = "buf11"} : memref<64x1xbf16, 2 : i32> 
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
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2x512x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2x512x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2x512x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<2x512x64xbf16>
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_74, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf288 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_75, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf286 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_73, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf282 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf289) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf291) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf290) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf288, %buf287) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf287, %buf288, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf290, %buf284, %buf283) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf283, %buf289) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf286, %buf289) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf291, %buf283, %buf284) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf284, %buf291) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf281 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf281 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf287, %buf288, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf281 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf290, %buf280, %buf279) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf279, %buf289) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf281 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf282, %buf289) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf291, %buf279, %buf280) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf280, %buf291) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf289 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf290 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_85 = memref.collapse_shape %buf291 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_5_74, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf275 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf273 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_70, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf269 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf276) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf278) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf277) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf275, %buf274) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf272 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf272 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf274, %buf275, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf272 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf277, %buf271, %buf270) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf270, %buf276) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf272 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf273, %buf276) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf278, %buf270, %buf271) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf271, %buf278) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf274, %buf275, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf277, %buf267, %buf266) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf266, %buf276) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf269, %buf276) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf278, %buf266, %buf267) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf267, %buf278) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf276 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf277 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_85 = memref.collapse_shape %buf278 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_5_71, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_68, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf262 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_69, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf260 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_67, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf256 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf263) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf265) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf264) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf262, %buf261) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf259 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf259 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf261, %buf262, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf259 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf264, %buf258, %buf257) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf257, %buf263) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf259 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf260, %buf263) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf265, %buf257, %buf258) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf258, %buf265) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf261, %buf262, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf264, %buf254, %buf253) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf253, %buf263) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf256, %buf263) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf265, %buf253, %buf254) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf254, %buf265) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf263 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf264 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_85 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_5_68, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_65, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf249 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_66, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf247 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_64, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf243 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf250) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf252) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf251) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf249, %buf248) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf246 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf246 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf248, %buf249, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf246 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf251, %buf245, %buf244) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf244, %buf250) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf246 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf247, %buf250) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf252, %buf244, %buf245) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf245, %buf252) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf242 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf242 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf248, %buf249, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf242 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf251, %buf241, %buf240) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf240, %buf250) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf242 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf243, %buf250) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf252, %buf240, %buf241) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf241, %buf252) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf250 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf251 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_85 = memref.collapse_shape %buf252 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_5_65, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf236 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_63, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_61, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf237) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf239) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf238) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf236, %buf235) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf233 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf233 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf235, %buf236, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf233 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf238, %buf232, %buf231) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf231, %buf237) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf233 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf234, %buf237) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf239, %buf231, %buf232) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf232, %buf239) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf229 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf229 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf235, %buf236, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf229 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf238, %buf228, %buf227) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf227, %buf237) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf229 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf230, %buf237) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf239, %buf227, %buf228) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf228, %buf239) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf225 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf224 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf238, %buf223) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf225, %buf238) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf225, %buf238, %buf222) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf223, %buf238, %buf221) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf222, %buf226) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf221, %buf237) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf237, %buf226) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf220) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf224, %buf222, %buf220) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf239, %buf221, %buf220) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf220, %buf224) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf238 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf224 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_4_62, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_59, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf216 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf214 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_58, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf210 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf217) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf219) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf218) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf216, %buf215) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf213 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf213 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf215, %buf216, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf213 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf218, %buf212, %buf211) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf211, %buf217) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf213 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf214, %buf217) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf219, %buf211, %buf212) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf212, %buf219) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf209 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf209 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf215, %buf216, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf209 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf218, %buf208, %buf207) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf207, %buf217) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf209 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf210, %buf217) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf219, %buf207, %buf208) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf208, %buf219) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf205 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf204 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf218, %buf203) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf205, %buf218) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf205, %buf218, %buf202) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf203, %buf218, %buf201) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf202, %buf206) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf201, %buf217) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf217, %buf206) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf200) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf204, %buf202, %buf200) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf219, %buf201, %buf200) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf200, %buf204) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf218 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf204 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_4_59, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf196 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf194 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_55, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf190 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf197) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf199) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf198) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf196, %buf195) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf193 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf193 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf195, %buf196, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf193 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf198, %buf192, %buf191) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf191, %buf197) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf193 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf194, %buf197) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf199, %buf191, %buf192) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf192, %buf199) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf189 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf189 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf195, %buf196, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf189 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf198, %buf188, %buf187) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf187, %buf197) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf189 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf190, %buf197) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf199, %buf187, %buf188) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf188, %buf199) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf186 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf184 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf198, %buf183) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf185, %buf198) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf185, %buf198, %buf182) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf183, %buf198, %buf181) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf182, %buf186) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf181, %buf197) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf197, %buf186) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf180) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf184, %buf182, %buf180) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf199, %buf181, %buf180) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf180, %buf184) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf186 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf198 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf184 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_4_56, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf176 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf174 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_52, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf170 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf177) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf179) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf178) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf176, %buf175) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf173 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf173 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf175, %buf176, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf173 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf178, %buf172, %buf171) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf171, %buf177) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf173 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf174, %buf177) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf179, %buf171, %buf172) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf172, %buf179) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf175, %buf176, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf178, %buf168, %buf167) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf167, %buf177) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf170, %buf177) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf179, %buf167, %buf168) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf168, %buf179) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf166 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf165 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf164 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf178, %buf163) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf165, %buf178) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf165, %buf178, %buf162) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf163, %buf178, %buf161) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf162, %buf166) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf161, %buf177) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf177, %buf166) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf160) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf164, %buf162, %buf160) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf179, %buf161, %buf160) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf160, %buf164) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf166 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf178 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf164 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_4_53, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf156 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf154 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf150 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf157) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf159) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf158) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf156, %buf155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf155, %buf156, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf158, %buf152, %buf151) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf151, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf154, %buf157) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf159, %buf151, %buf152) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf152, %buf159) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf149 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf149 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf155, %buf156, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf149 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf158, %buf148, %buf147) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf147, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf149 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf150, %buf157) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf159, %buf147, %buf148) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf148, %buf159) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf146 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf145 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf144 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf158, %buf143) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf145, %buf158) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf145, %buf158, %buf142) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf143, %buf158, %buf141) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf142, %buf146) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf141, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf157, %buf146) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf140) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf144, %buf142, %buf140) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf159, %buf141, %buf140) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf140, %buf144) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf146 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf158 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf144 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_3_50, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf136 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf134 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf130 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf137) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf139) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf138) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf136, %buf135) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf133 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf133 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf135, %buf136, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf133 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf138, %buf132, %buf131) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf131, %buf137) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf133 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf134, %buf137) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf139, %buf131, %buf132) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf132, %buf139) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf129 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf129 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf135, %buf136, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf129 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf138, %buf128, %buf127) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf127, %buf137) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf129 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf130, %buf137) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf139, %buf127, %buf128) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf128, %buf139) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf126 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf125 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf124 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf138, %buf123) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf125, %buf138) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf125, %buf138, %buf122) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf123, %buf138, %buf121) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf122, %buf126) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf121, %buf137) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf137, %buf126) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf120) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf124, %buf122, %buf120) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf139, %buf121, %buf120) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf120, %buf124) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf126 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf138 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf124 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_3_47, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_44, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf116 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf114 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf110 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf117) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf119) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf118) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf116, %buf115) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf113 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf113 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf115, %buf116, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf113 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf118, %buf112, %buf111) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf111, %buf117) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf113 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf114, %buf117) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf119, %buf111, %buf112) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf112, %buf119) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf109 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf109 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf115, %buf116, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf109 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf118, %buf108, %buf107) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf107, %buf117) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf109 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf110, %buf117) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf119, %buf107, %buf108) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf108, %buf119) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf106 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf105 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf104 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf118, %buf103) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf105, %buf118) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf105, %buf118, %buf102) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf103, %buf118, %buf101) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf102, %buf106) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf101, %buf117) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf117, %buf106) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf100) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf104, %buf102, %buf100) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf119, %buf101, %buf100) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf100, %buf104) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf106 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf118 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf104 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_3_44, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf96 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf94 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_40, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf97) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf99) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf98) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf96, %buf95) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf93 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf93 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf95, %buf96, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf93 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf98, %buf92, %buf91) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf91, %buf97) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf93 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf94, %buf97) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf99, %buf91, %buf92) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf92, %buf99) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf95, %buf96, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf98, %buf88, %buf87) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf87, %buf97) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf90, %buf97) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf99, %buf87, %buf88) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf88, %buf99) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf86 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf85 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf84 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf98, %buf83) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf85, %buf98) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf85, %buf98, %buf82) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf83, %buf98, %buf81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf82, %buf86) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf81, %buf97) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf97, %buf86) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf80) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf84, %buf82, %buf80) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf99, %buf81, %buf80) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf80, %buf84) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf86 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf98 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf84 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_3_41, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf66 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_35, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf70 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf77) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf79) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf78) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf76, %buf75) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf75, %buf76, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf78, %buf72, %buf71) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf71, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf74, %buf77) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf79, %buf71, %buf72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf72, %buf79) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf69 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf69 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf75, %buf76, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf69 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf78, %buf68, %buf67) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf67, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf69 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf70, %buf77) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf79, %buf67, %buf68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf68, %buf79) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf66 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf65 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf64 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf78, %buf63) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf65, %buf78) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf65, %buf78, %buf62) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf63, %buf78, %buf61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf62, %buf66) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf61, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf77, %buf66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf60) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf64, %buf62, %buf60) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf79, %buf61, %buf60) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf60, %buf64) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf64, %buf66) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_39, Release, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf46 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf56 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf54 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf50 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf57) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf59) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf58) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf56, %buf55) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf53 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf53 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf55, %buf56, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf53 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf58, %buf52, %buf51) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf51, %buf57) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf53 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf54, %buf57) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf59, %buf51, %buf52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf52, %buf59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf49 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf49 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf55, %buf56, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf49 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf58, %buf48, %buf47) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf47, %buf57) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf49 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf50, %buf57) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf59, %buf47, %buf48) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf48, %buf59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf46 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf45 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf44 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf58, %buf43) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf45, %buf58) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf45, %buf58, %buf42) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf43, %buf58, %buf41) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf42, %buf46) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf41, %buf57) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf57, %buf46) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf40) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf44, %buf42, %buf40) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf59, %buf41, %buf40) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf40, %buf44) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf44, %buf46) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf36 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf34 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_25, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf37) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf39) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf38) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf36, %buf35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf33 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf33 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf35, %buf36, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf33 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf38, %buf32, %buf31) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf31, %buf37) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf33 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf34, %buf37) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf39, %buf31, %buf32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf32, %buf39) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf29 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf29 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf35, %buf36, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf29 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf38, %buf28, %buf27) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf27, %buf37) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf29 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf30, %buf37) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf39, %buf27, %buf28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf28, %buf39) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf26 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf25 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf24 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf38, %buf23) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf25, %buf38) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf25, %buf38, %buf22) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf23, %buf38, %buf21) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf22, %buf26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf21, %buf37) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf37, %buf26) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf20) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf24, %buf22, %buf20) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf39, %buf21, %buf20) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf20, %buf24) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf24, %buf26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_29, Release, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
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
      aie.dma_bd(%buf16 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf17) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf19) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf18) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf16, %buf15) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf13 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf13 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf15, %buf16, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf13 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf18, %buf12, %buf11) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf11, %buf17) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf13 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf14, %buf17) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf19, %buf11, %buf12) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf12, %buf19) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf15, %buf16, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf18, %buf8, %buf7) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf7, %buf17) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf10, %buf17) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf19, %buf7, %buf8) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf8, %buf19) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf6 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf5 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf4 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf18, %buf3) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf5, %buf18) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf5, %buf18, %buf2) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf3, %buf18, %buf1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf2, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf1, %buf17) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf17, %buf6) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf0) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf4, %buf2, %buf0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf19, %buf1, %buf0) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf0, %buf4) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf4, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_24, Release, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
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
      aie.use_lock(%lock_0_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf303 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf299 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_16, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_0_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf299 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_17, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_15, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf303 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_19, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf302 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf297 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_11, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf296 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_1_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf297 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_12, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf296 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_10, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf302 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_14, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf301 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf295 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_6, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf294 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_2_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf295 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_7, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf294 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf301 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf300 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf293 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf292 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf293 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf292 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_3_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf300 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_4, Release, 1)
      aie.next_bd ^bb10
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
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 2 : i32}
    %lock_0_2_20 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_21 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_22 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_23 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_24 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 2 : i32}
    %lock_1_2_25 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_26 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_27 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_28 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_29 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 2 : i32}
    %lock_2_2_30 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_31 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_32 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_33 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_34 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 2 : i32}
    %lock_3_2_35 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_36 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_37 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_38 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_39 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 2 : i32}
    %lock_0_3_40 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_41 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_42 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 3) {init = 2 : i32}
    %lock_1_3_43 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_44 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_45 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 3) {init = 2 : i32}
    %lock_2_3_46 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_47 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_48 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 3) {init = 2 : i32}
    %lock_3_3_49 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_50 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_51 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 2 : i32}
    %lock_0_4_52 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_53 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_54 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 3) {init = 2 : i32}
    %lock_1_4_55 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_56 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_57 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 3) {init = 2 : i32}
    %lock_2_4_58 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_59 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_60 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 3) {init = 2 : i32}
    %lock_3_4_61 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_62 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_63 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 2 : i32}
    %lock_0_5_64 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_65 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_66 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 3) {init = 2 : i32}
    %lock_1_5_67 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_68 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_69 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 3) {init = 2 : i32}
    %lock_2_5_70 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_71 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_72 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 3) {init = 2 : i32}
    %lock_3_5_73 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_74 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_75 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf607 = aie.buffer(%mem_tile_0_1) {sym_name = "buf607"} : memref<64x64xbf16, 1 : i32> 
    %buf606 = aie.buffer(%mem_tile_1_1) {sym_name = "buf606"} : memref<64x64xbf16, 1 : i32> 
    %buf605 = aie.buffer(%mem_tile_2_1) {sym_name = "buf605"} : memref<64x64xbf16, 1 : i32> 
    %buf604 = aie.buffer(%mem_tile_3_1) {sym_name = "buf604"} : memref<64x64xbf16, 1 : i32> 
    %buf603 = aie.buffer(%mem_tile_0_1) {sym_name = "buf603"} : memref<64x64xbf16, 1 : i32> 
    %buf602 = aie.buffer(%mem_tile_0_1) {sym_name = "buf602"} : memref<64x64xbf16, 1 : i32> 
    %buf601 = aie.buffer(%mem_tile_1_1) {sym_name = "buf601"} : memref<64x64xbf16, 1 : i32> 
    %buf600 = aie.buffer(%mem_tile_1_1) {sym_name = "buf600"} : memref<64x64xbf16, 1 : i32> 
    %buf599 = aie.buffer(%mem_tile_2_1) {sym_name = "buf599"} : memref<64x64xbf16, 1 : i32> 
    %buf598 = aie.buffer(%mem_tile_2_1) {sym_name = "buf598"} : memref<64x64xbf16, 1 : i32> 
    %buf597 = aie.buffer(%mem_tile_3_1) {sym_name = "buf597"} : memref<64x64xbf16, 1 : i32> 
    %buf596 = aie.buffer(%mem_tile_3_1) {sym_name = "buf596"} : memref<64x64xbf16, 1 : i32> 
    %buf595 = aie.buffer(%tile_3_5) {sym_name = "buf595"} : memref<64x1xbf16, 2 : i32> 
    %buf594 = aie.buffer(%tile_3_5) {sym_name = "buf594"} : memref<64x1xbf16, 2 : i32> 
    %buf593 = aie.buffer(%tile_3_5) {sym_name = "buf593"} : memref<64x64xbf16, 2 : i32> 
    %buf592 = aie.buffer(%tile_3_5) {sym_name = "buf592"} : memref<64x64xbf16, 2 : i32> 
    %buf591 = aie.buffer(%tile_3_5) {sym_name = "buf591"} : memref<64x64xbf16, 2 : i32> 
    %buf590 = aie.buffer(%tile_3_5) {sym_name = "buf590"} : memref<64x64xbf16, 2 : i32> 
    %buf589 = aie.buffer(%tile_3_5) {sym_name = "buf589"} : memref<64x64xbf16, 2 : i32> 
    %buf588 = aie.buffer(%tile_3_5) {sym_name = "buf588"} : memref<64x1xbf16, 2 : i32> 
    %buf587 = aie.buffer(%tile_3_5) {sym_name = "buf587"} : memref<64x1xbf16, 2 : i32> 
    %buf586 = aie.buffer(%tile_3_5) {sym_name = "buf586"} : memref<64x64xbf16, 2 : i32> 
    %buf585 = aie.buffer(%tile_3_5) {sym_name = "buf585"} : memref<64x64xbf16, 2 : i32> 
    %buf584 = aie.buffer(%tile_3_5) {sym_name = "buf584"} : memref<64x1xbf16, 2 : i32> 
    %buf583 = aie.buffer(%tile_3_5) {sym_name = "buf583"} : memref<64x1xbf16, 2 : i32> 
    %buf582 = aie.buffer(%tile_2_5) {sym_name = "buf582"} : memref<64x1xbf16, 2 : i32> 
    %buf581 = aie.buffer(%tile_2_5) {sym_name = "buf581"} : memref<64x1xbf16, 2 : i32> 
    %buf580 = aie.buffer(%tile_2_5) {sym_name = "buf580"} : memref<64x64xbf16, 2 : i32> 
    %buf579 = aie.buffer(%tile_2_5) {sym_name = "buf579"} : memref<64x64xbf16, 2 : i32> 
    %buf578 = aie.buffer(%tile_2_5) {sym_name = "buf578"} : memref<64x64xbf16, 2 : i32> 
    %buf577 = aie.buffer(%tile_2_5) {sym_name = "buf577"} : memref<64x64xbf16, 2 : i32> 
    %buf576 = aie.buffer(%tile_2_5) {sym_name = "buf576"} : memref<64x64xbf16, 2 : i32> 
    %buf575 = aie.buffer(%tile_2_5) {sym_name = "buf575"} : memref<64x1xbf16, 2 : i32> 
    %buf574 = aie.buffer(%tile_2_5) {sym_name = "buf574"} : memref<64x1xbf16, 2 : i32> 
    %buf573 = aie.buffer(%tile_2_5) {sym_name = "buf573"} : memref<64x64xbf16, 2 : i32> 
    %buf572 = aie.buffer(%tile_2_5) {sym_name = "buf572"} : memref<64x64xbf16, 2 : i32> 
    %buf571 = aie.buffer(%tile_2_5) {sym_name = "buf571"} : memref<64x1xbf16, 2 : i32> 
    %buf570 = aie.buffer(%tile_2_5) {sym_name = "buf570"} : memref<64x1xbf16, 2 : i32> 
    %buf569 = aie.buffer(%tile_1_5) {sym_name = "buf569"} : memref<64x1xbf16, 2 : i32> 
    %buf568 = aie.buffer(%tile_1_5) {sym_name = "buf568"} : memref<64x1xbf16, 2 : i32> 
    %buf567 = aie.buffer(%tile_1_5) {sym_name = "buf567"} : memref<64x64xbf16, 2 : i32> 
    %buf566 = aie.buffer(%tile_1_5) {sym_name = "buf566"} : memref<64x64xbf16, 2 : i32> 
    %buf565 = aie.buffer(%tile_1_5) {sym_name = "buf565"} : memref<64x64xbf16, 2 : i32> 
    %buf564 = aie.buffer(%tile_1_5) {sym_name = "buf564"} : memref<64x64xbf16, 2 : i32> 
    %buf563 = aie.buffer(%tile_1_5) {sym_name = "buf563"} : memref<64x64xbf16, 2 : i32> 
    %buf562 = aie.buffer(%tile_1_5) {sym_name = "buf562"} : memref<64x1xbf16, 2 : i32> 
    %buf561 = aie.buffer(%tile_1_5) {sym_name = "buf561"} : memref<64x1xbf16, 2 : i32> 
    %buf560 = aie.buffer(%tile_1_5) {sym_name = "buf560"} : memref<64x64xbf16, 2 : i32> 
    %buf559 = aie.buffer(%tile_1_5) {sym_name = "buf559"} : memref<64x64xbf16, 2 : i32> 
    %buf558 = aie.buffer(%tile_1_5) {sym_name = "buf558"} : memref<64x1xbf16, 2 : i32> 
    %buf557 = aie.buffer(%tile_1_5) {sym_name = "buf557"} : memref<64x1xbf16, 2 : i32> 
    %buf556 = aie.buffer(%tile_0_5) {sym_name = "buf556"} : memref<64x1xbf16, 2 : i32> 
    %buf555 = aie.buffer(%tile_0_5) {sym_name = "buf555"} : memref<64x1xbf16, 2 : i32> 
    %buf554 = aie.buffer(%tile_0_5) {sym_name = "buf554"} : memref<64x64xbf16, 2 : i32> 
    %buf553 = aie.buffer(%tile_0_5) {sym_name = "buf553"} : memref<64x64xbf16, 2 : i32> 
    %buf552 = aie.buffer(%tile_0_5) {sym_name = "buf552"} : memref<64x64xbf16, 2 : i32> 
    %buf551 = aie.buffer(%tile_0_5) {sym_name = "buf551"} : memref<64x64xbf16, 2 : i32> 
    %buf550 = aie.buffer(%tile_0_5) {sym_name = "buf550"} : memref<64x64xbf16, 2 : i32> 
    %buf549 = aie.buffer(%tile_0_5) {sym_name = "buf549"} : memref<64x1xbf16, 2 : i32> 
    %buf548 = aie.buffer(%tile_0_5) {sym_name = "buf548"} : memref<64x1xbf16, 2 : i32> 
    %buf547 = aie.buffer(%tile_0_5) {sym_name = "buf547"} : memref<64x64xbf16, 2 : i32> 
    %buf546 = aie.buffer(%tile_0_5) {sym_name = "buf546"} : memref<64x64xbf16, 2 : i32> 
    %buf545 = aie.buffer(%tile_0_5) {sym_name = "buf545"} : memref<64x1xbf16, 2 : i32> 
    %buf544 = aie.buffer(%tile_0_5) {sym_name = "buf544"} : memref<64x1xbf16, 2 : i32> 
    %buf543 = aie.buffer(%tile_3_4) {sym_name = "buf543"} : memref<64x1xbf16, 2 : i32> 
    %buf542 = aie.buffer(%tile_3_4) {sym_name = "buf542"} : memref<64x1xbf16, 2 : i32> 
    %buf541 = aie.buffer(%tile_3_4) {sym_name = "buf541"} : memref<64x64xbf16, 2 : i32> 
    %buf540 = aie.buffer(%tile_3_4) {sym_name = "buf540"} : memref<64x64xbf16, 2 : i32> 
    %buf539 = aie.buffer(%tile_3_4) {sym_name = "buf539"} : memref<64x64xbf16, 2 : i32> 
    %buf538 = aie.buffer(%tile_3_4) {sym_name = "buf538"} : memref<64x64xbf16, 2 : i32> 
    %buf537 = aie.buffer(%tile_3_4) {sym_name = "buf537"} : memref<64x64xbf16, 2 : i32> 
    %buf536 = aie.buffer(%tile_3_4) {sym_name = "buf536"} : memref<64x1xbf16, 2 : i32> 
    %buf535 = aie.buffer(%tile_3_4) {sym_name = "buf535"} : memref<64x1xbf16, 2 : i32> 
    %buf534 = aie.buffer(%tile_3_4) {sym_name = "buf534"} : memref<64x64xbf16, 2 : i32> 
    %buf533 = aie.buffer(%tile_3_4) {sym_name = "buf533"} : memref<64x64xbf16, 2 : i32> 
    %buf532 = aie.buffer(%tile_3_4) {sym_name = "buf532"} : memref<64x1xbf16, 2 : i32> 
    %buf531 = aie.buffer(%tile_3_4) {sym_name = "buf531"} : memref<64x1xbf16, 2 : i32> 
    %buf530 = aie.buffer(%tile_3_4) {sym_name = "buf530"} : memref<64x64xbf16, 2 : i32> 
    %buf529 = aie.buffer(%tile_3_4) {sym_name = "buf529"} : memref<64x1xbf16, 2 : i32> 
    %buf528 = aie.buffer(%tile_3_4) {sym_name = "buf528"} : memref<64x1xbf16, 2 : i32> 
    %buf527 = aie.buffer(%tile_3_4) {sym_name = "buf527"} : memref<64x1xbf16, 2 : i32> 
    %buf526 = aie.buffer(%tile_3_4) {sym_name = "buf526"} : memref<64x1xbf16, 2 : i32> 
    %buf525 = aie.buffer(%tile_3_4) {sym_name = "buf525"} : memref<64x1xbf16, 2 : i32> 
    %buf524 = aie.buffer(%tile_3_4) {sym_name = "buf524"} : memref<64x1xbf16, 2 : i32> 
    %buf523 = aie.buffer(%tile_2_4) {sym_name = "buf523"} : memref<64x1xbf16, 2 : i32> 
    %buf522 = aie.buffer(%tile_2_4) {sym_name = "buf522"} : memref<64x1xbf16, 2 : i32> 
    %buf521 = aie.buffer(%tile_2_4) {sym_name = "buf521"} : memref<64x64xbf16, 2 : i32> 
    %buf520 = aie.buffer(%tile_2_4) {sym_name = "buf520"} : memref<64x64xbf16, 2 : i32> 
    %buf519 = aie.buffer(%tile_2_4) {sym_name = "buf519"} : memref<64x64xbf16, 2 : i32> 
    %buf518 = aie.buffer(%tile_2_4) {sym_name = "buf518"} : memref<64x64xbf16, 2 : i32> 
    %buf517 = aie.buffer(%tile_2_4) {sym_name = "buf517"} : memref<64x64xbf16, 2 : i32> 
    %buf516 = aie.buffer(%tile_2_4) {sym_name = "buf516"} : memref<64x1xbf16, 2 : i32> 
    %buf515 = aie.buffer(%tile_2_4) {sym_name = "buf515"} : memref<64x1xbf16, 2 : i32> 
    %buf514 = aie.buffer(%tile_2_4) {sym_name = "buf514"} : memref<64x64xbf16, 2 : i32> 
    %buf513 = aie.buffer(%tile_2_4) {sym_name = "buf513"} : memref<64x64xbf16, 2 : i32> 
    %buf512 = aie.buffer(%tile_2_4) {sym_name = "buf512"} : memref<64x1xbf16, 2 : i32> 
    %buf511 = aie.buffer(%tile_2_4) {sym_name = "buf511"} : memref<64x1xbf16, 2 : i32> 
    %buf510 = aie.buffer(%tile_2_4) {sym_name = "buf510"} : memref<64x64xbf16, 2 : i32> 
    %buf509 = aie.buffer(%tile_2_4) {sym_name = "buf509"} : memref<64x1xbf16, 2 : i32> 
    %buf508 = aie.buffer(%tile_2_4) {sym_name = "buf508"} : memref<64x1xbf16, 2 : i32> 
    %buf507 = aie.buffer(%tile_2_4) {sym_name = "buf507"} : memref<64x1xbf16, 2 : i32> 
    %buf506 = aie.buffer(%tile_2_4) {sym_name = "buf506"} : memref<64x1xbf16, 2 : i32> 
    %buf505 = aie.buffer(%tile_2_4) {sym_name = "buf505"} : memref<64x1xbf16, 2 : i32> 
    %buf504 = aie.buffer(%tile_2_4) {sym_name = "buf504"} : memref<64x1xbf16, 2 : i32> 
    %buf503 = aie.buffer(%tile_1_4) {sym_name = "buf503"} : memref<64x1xbf16, 2 : i32> 
    %buf502 = aie.buffer(%tile_1_4) {sym_name = "buf502"} : memref<64x1xbf16, 2 : i32> 
    %buf501 = aie.buffer(%tile_1_4) {sym_name = "buf501"} : memref<64x64xbf16, 2 : i32> 
    %buf500 = aie.buffer(%tile_1_4) {sym_name = "buf500"} : memref<64x64xbf16, 2 : i32> 
    %buf499 = aie.buffer(%tile_1_4) {sym_name = "buf499"} : memref<64x64xbf16, 2 : i32> 
    %buf498 = aie.buffer(%tile_1_4) {sym_name = "buf498"} : memref<64x64xbf16, 2 : i32> 
    %buf497 = aie.buffer(%tile_1_4) {sym_name = "buf497"} : memref<64x64xbf16, 2 : i32> 
    %buf496 = aie.buffer(%tile_1_4) {sym_name = "buf496"} : memref<64x1xbf16, 2 : i32> 
    %buf495 = aie.buffer(%tile_1_4) {sym_name = "buf495"} : memref<64x1xbf16, 2 : i32> 
    %buf494 = aie.buffer(%tile_1_4) {sym_name = "buf494"} : memref<64x64xbf16, 2 : i32> 
    %buf493 = aie.buffer(%tile_1_4) {sym_name = "buf493"} : memref<64x64xbf16, 2 : i32> 
    %buf492 = aie.buffer(%tile_1_4) {sym_name = "buf492"} : memref<64x1xbf16, 2 : i32> 
    %buf491 = aie.buffer(%tile_1_4) {sym_name = "buf491"} : memref<64x1xbf16, 2 : i32> 
    %buf490 = aie.buffer(%tile_1_4) {sym_name = "buf490"} : memref<64x64xbf16, 2 : i32> 
    %buf489 = aie.buffer(%tile_1_4) {sym_name = "buf489"} : memref<64x1xbf16, 2 : i32> 
    %buf488 = aie.buffer(%tile_1_4) {sym_name = "buf488"} : memref<64x1xbf16, 2 : i32> 
    %buf487 = aie.buffer(%tile_1_4) {sym_name = "buf487"} : memref<64x1xbf16, 2 : i32> 
    %buf486 = aie.buffer(%tile_1_4) {sym_name = "buf486"} : memref<64x1xbf16, 2 : i32> 
    %buf485 = aie.buffer(%tile_1_4) {sym_name = "buf485"} : memref<64x1xbf16, 2 : i32> 
    %buf484 = aie.buffer(%tile_1_4) {sym_name = "buf484"} : memref<64x1xbf16, 2 : i32> 
    %buf483 = aie.buffer(%tile_0_4) {sym_name = "buf483"} : memref<64x1xbf16, 2 : i32> 
    %buf482 = aie.buffer(%tile_0_4) {sym_name = "buf482"} : memref<64x1xbf16, 2 : i32> 
    %buf481 = aie.buffer(%tile_0_4) {sym_name = "buf481"} : memref<64x64xbf16, 2 : i32> 
    %buf480 = aie.buffer(%tile_0_4) {sym_name = "buf480"} : memref<64x64xbf16, 2 : i32> 
    %buf479 = aie.buffer(%tile_0_4) {sym_name = "buf479"} : memref<64x64xbf16, 2 : i32> 
    %buf478 = aie.buffer(%tile_0_4) {sym_name = "buf478"} : memref<64x64xbf16, 2 : i32> 
    %buf477 = aie.buffer(%tile_0_4) {sym_name = "buf477"} : memref<64x64xbf16, 2 : i32> 
    %buf476 = aie.buffer(%tile_0_4) {sym_name = "buf476"} : memref<64x1xbf16, 2 : i32> 
    %buf475 = aie.buffer(%tile_0_4) {sym_name = "buf475"} : memref<64x1xbf16, 2 : i32> 
    %buf474 = aie.buffer(%tile_0_4) {sym_name = "buf474"} : memref<64x64xbf16, 2 : i32> 
    %buf473 = aie.buffer(%tile_0_4) {sym_name = "buf473"} : memref<64x64xbf16, 2 : i32> 
    %buf472 = aie.buffer(%tile_0_4) {sym_name = "buf472"} : memref<64x1xbf16, 2 : i32> 
    %buf471 = aie.buffer(%tile_0_4) {sym_name = "buf471"} : memref<64x1xbf16, 2 : i32> 
    %buf470 = aie.buffer(%tile_0_4) {sym_name = "buf470"} : memref<64x64xbf16, 2 : i32> 
    %buf469 = aie.buffer(%tile_0_4) {sym_name = "buf469"} : memref<64x1xbf16, 2 : i32> 
    %buf468 = aie.buffer(%tile_0_4) {sym_name = "buf468"} : memref<64x1xbf16, 2 : i32> 
    %buf467 = aie.buffer(%tile_0_4) {sym_name = "buf467"} : memref<64x1xbf16, 2 : i32> 
    %buf466 = aie.buffer(%tile_0_4) {sym_name = "buf466"} : memref<64x1xbf16, 2 : i32> 
    %buf465 = aie.buffer(%tile_0_4) {sym_name = "buf465"} : memref<64x1xbf16, 2 : i32> 
    %buf464 = aie.buffer(%tile_0_4) {sym_name = "buf464"} : memref<64x1xbf16, 2 : i32> 
    %buf463 = aie.buffer(%tile_3_3) {sym_name = "buf463"} : memref<64x1xbf16, 2 : i32> 
    %buf462 = aie.buffer(%tile_3_3) {sym_name = "buf462"} : memref<64x1xbf16, 2 : i32> 
    %buf461 = aie.buffer(%tile_3_3) {sym_name = "buf461"} : memref<64x64xbf16, 2 : i32> 
    %buf460 = aie.buffer(%tile_3_3) {sym_name = "buf460"} : memref<64x64xbf16, 2 : i32> 
    %buf459 = aie.buffer(%tile_3_3) {sym_name = "buf459"} : memref<64x64xbf16, 2 : i32> 
    %buf458 = aie.buffer(%tile_3_3) {sym_name = "buf458"} : memref<64x64xbf16, 2 : i32> 
    %buf457 = aie.buffer(%tile_3_3) {sym_name = "buf457"} : memref<64x64xbf16, 2 : i32> 
    %buf456 = aie.buffer(%tile_3_3) {sym_name = "buf456"} : memref<64x1xbf16, 2 : i32> 
    %buf455 = aie.buffer(%tile_3_3) {sym_name = "buf455"} : memref<64x1xbf16, 2 : i32> 
    %buf454 = aie.buffer(%tile_3_3) {sym_name = "buf454"} : memref<64x64xbf16, 2 : i32> 
    %buf453 = aie.buffer(%tile_3_3) {sym_name = "buf453"} : memref<64x64xbf16, 2 : i32> 
    %buf452 = aie.buffer(%tile_3_3) {sym_name = "buf452"} : memref<64x1xbf16, 2 : i32> 
    %buf451 = aie.buffer(%tile_3_3) {sym_name = "buf451"} : memref<64x1xbf16, 2 : i32> 
    %buf450 = aie.buffer(%tile_3_3) {sym_name = "buf450"} : memref<64x64xbf16, 2 : i32> 
    %buf449 = aie.buffer(%tile_3_3) {sym_name = "buf449"} : memref<64x1xbf16, 2 : i32> 
    %buf448 = aie.buffer(%tile_3_3) {sym_name = "buf448"} : memref<64x1xbf16, 2 : i32> 
    %buf447 = aie.buffer(%tile_3_3) {sym_name = "buf447"} : memref<64x1xbf16, 2 : i32> 
    %buf446 = aie.buffer(%tile_3_3) {sym_name = "buf446"} : memref<64x1xbf16, 2 : i32> 
    %buf445 = aie.buffer(%tile_3_3) {sym_name = "buf445"} : memref<64x1xbf16, 2 : i32> 
    %buf444 = aie.buffer(%tile_3_3) {sym_name = "buf444"} : memref<64x1xbf16, 2 : i32> 
    %buf443 = aie.buffer(%tile_2_3) {sym_name = "buf443"} : memref<64x1xbf16, 2 : i32> 
    %buf442 = aie.buffer(%tile_2_3) {sym_name = "buf442"} : memref<64x1xbf16, 2 : i32> 
    %buf441 = aie.buffer(%tile_2_3) {sym_name = "buf441"} : memref<64x64xbf16, 2 : i32> 
    %buf440 = aie.buffer(%tile_2_3) {sym_name = "buf440"} : memref<64x64xbf16, 2 : i32> 
    %buf439 = aie.buffer(%tile_2_3) {sym_name = "buf439"} : memref<64x64xbf16, 2 : i32> 
    %buf438 = aie.buffer(%tile_2_3) {sym_name = "buf438"} : memref<64x64xbf16, 2 : i32> 
    %buf437 = aie.buffer(%tile_2_3) {sym_name = "buf437"} : memref<64x64xbf16, 2 : i32> 
    %buf436 = aie.buffer(%tile_2_3) {sym_name = "buf436"} : memref<64x1xbf16, 2 : i32> 
    %buf435 = aie.buffer(%tile_2_3) {sym_name = "buf435"} : memref<64x1xbf16, 2 : i32> 
    %buf434 = aie.buffer(%tile_2_3) {sym_name = "buf434"} : memref<64x64xbf16, 2 : i32> 
    %buf433 = aie.buffer(%tile_2_3) {sym_name = "buf433"} : memref<64x64xbf16, 2 : i32> 
    %buf432 = aie.buffer(%tile_2_3) {sym_name = "buf432"} : memref<64x1xbf16, 2 : i32> 
    %buf431 = aie.buffer(%tile_2_3) {sym_name = "buf431"} : memref<64x1xbf16, 2 : i32> 
    %buf430 = aie.buffer(%tile_2_3) {sym_name = "buf430"} : memref<64x64xbf16, 2 : i32> 
    %buf429 = aie.buffer(%tile_2_3) {sym_name = "buf429"} : memref<64x1xbf16, 2 : i32> 
    %buf428 = aie.buffer(%tile_2_3) {sym_name = "buf428"} : memref<64x1xbf16, 2 : i32> 
    %buf427 = aie.buffer(%tile_2_3) {sym_name = "buf427"} : memref<64x1xbf16, 2 : i32> 
    %buf426 = aie.buffer(%tile_2_3) {sym_name = "buf426"} : memref<64x1xbf16, 2 : i32> 
    %buf425 = aie.buffer(%tile_2_3) {sym_name = "buf425"} : memref<64x1xbf16, 2 : i32> 
    %buf424 = aie.buffer(%tile_2_3) {sym_name = "buf424"} : memref<64x1xbf16, 2 : i32> 
    %buf423 = aie.buffer(%tile_1_3) {sym_name = "buf423"} : memref<64x1xbf16, 2 : i32> 
    %buf422 = aie.buffer(%tile_1_3) {sym_name = "buf422"} : memref<64x1xbf16, 2 : i32> 
    %buf421 = aie.buffer(%tile_1_3) {sym_name = "buf421"} : memref<64x64xbf16, 2 : i32> 
    %buf420 = aie.buffer(%tile_1_3) {sym_name = "buf420"} : memref<64x64xbf16, 2 : i32> 
    %buf419 = aie.buffer(%tile_1_3) {sym_name = "buf419"} : memref<64x64xbf16, 2 : i32> 
    %buf418 = aie.buffer(%tile_1_3) {sym_name = "buf418"} : memref<64x64xbf16, 2 : i32> 
    %buf417 = aie.buffer(%tile_1_3) {sym_name = "buf417"} : memref<64x64xbf16, 2 : i32> 
    %buf416 = aie.buffer(%tile_1_3) {sym_name = "buf416"} : memref<64x1xbf16, 2 : i32> 
    %buf415 = aie.buffer(%tile_1_3) {sym_name = "buf415"} : memref<64x1xbf16, 2 : i32> 
    %buf414 = aie.buffer(%tile_1_3) {sym_name = "buf414"} : memref<64x64xbf16, 2 : i32> 
    %buf413 = aie.buffer(%tile_1_3) {sym_name = "buf413"} : memref<64x64xbf16, 2 : i32> 
    %buf412 = aie.buffer(%tile_1_3) {sym_name = "buf412"} : memref<64x1xbf16, 2 : i32> 
    %buf411 = aie.buffer(%tile_1_3) {sym_name = "buf411"} : memref<64x1xbf16, 2 : i32> 
    %buf410 = aie.buffer(%tile_1_3) {sym_name = "buf410"} : memref<64x64xbf16, 2 : i32> 
    %buf409 = aie.buffer(%tile_1_3) {sym_name = "buf409"} : memref<64x1xbf16, 2 : i32> 
    %buf408 = aie.buffer(%tile_1_3) {sym_name = "buf408"} : memref<64x1xbf16, 2 : i32> 
    %buf407 = aie.buffer(%tile_1_3) {sym_name = "buf407"} : memref<64x1xbf16, 2 : i32> 
    %buf406 = aie.buffer(%tile_1_3) {sym_name = "buf406"} : memref<64x1xbf16, 2 : i32> 
    %buf405 = aie.buffer(%tile_1_3) {sym_name = "buf405"} : memref<64x1xbf16, 2 : i32> 
    %buf404 = aie.buffer(%tile_1_3) {sym_name = "buf404"} : memref<64x1xbf16, 2 : i32> 
    %buf403 = aie.buffer(%tile_0_3) {sym_name = "buf403"} : memref<64x1xbf16, 2 : i32> 
    %buf402 = aie.buffer(%tile_0_3) {sym_name = "buf402"} : memref<64x1xbf16, 2 : i32> 
    %buf401 = aie.buffer(%tile_0_3) {sym_name = "buf401"} : memref<64x64xbf16, 2 : i32> 
    %buf400 = aie.buffer(%tile_0_3) {sym_name = "buf400"} : memref<64x64xbf16, 2 : i32> 
    %buf399 = aie.buffer(%tile_0_3) {sym_name = "buf399"} : memref<64x64xbf16, 2 : i32> 
    %buf398 = aie.buffer(%tile_0_3) {sym_name = "buf398"} : memref<64x64xbf16, 2 : i32> 
    %buf397 = aie.buffer(%tile_0_3) {sym_name = "buf397"} : memref<64x64xbf16, 2 : i32> 
    %buf396 = aie.buffer(%tile_0_3) {sym_name = "buf396"} : memref<64x1xbf16, 2 : i32> 
    %buf395 = aie.buffer(%tile_0_3) {sym_name = "buf395"} : memref<64x1xbf16, 2 : i32> 
    %buf394 = aie.buffer(%tile_0_3) {sym_name = "buf394"} : memref<64x64xbf16, 2 : i32> 
    %buf393 = aie.buffer(%tile_0_3) {sym_name = "buf393"} : memref<64x64xbf16, 2 : i32> 
    %buf392 = aie.buffer(%tile_0_3) {sym_name = "buf392"} : memref<64x1xbf16, 2 : i32> 
    %buf391 = aie.buffer(%tile_0_3) {sym_name = "buf391"} : memref<64x1xbf16, 2 : i32> 
    %buf390 = aie.buffer(%tile_0_3) {sym_name = "buf390"} : memref<64x64xbf16, 2 : i32> 
    %buf389 = aie.buffer(%tile_0_3) {sym_name = "buf389"} : memref<64x1xbf16, 2 : i32> 
    %buf388 = aie.buffer(%tile_0_3) {sym_name = "buf388"} : memref<64x1xbf16, 2 : i32> 
    %buf387 = aie.buffer(%tile_0_3) {sym_name = "buf387"} : memref<64x1xbf16, 2 : i32> 
    %buf386 = aie.buffer(%tile_0_3) {sym_name = "buf386"} : memref<64x1xbf16, 2 : i32> 
    %buf385 = aie.buffer(%tile_0_3) {sym_name = "buf385"} : memref<64x1xbf16, 2 : i32> 
    %buf384 = aie.buffer(%tile_0_3) {sym_name = "buf384"} : memref<64x1xbf16, 2 : i32> 
    %buf383 = aie.buffer(%tile_3_2) {sym_name = "buf383"} : memref<64x1xbf16, 2 : i32> 
    %buf382 = aie.buffer(%tile_3_2) {sym_name = "buf382"} : memref<64x1xbf16, 2 : i32> 
    %buf381 = aie.buffer(%tile_3_2) {sym_name = "buf381"} : memref<64x64xbf16, 2 : i32> 
    %buf380 = aie.buffer(%tile_3_2) {sym_name = "buf380"} : memref<64x64xbf16, 2 : i32> 
    %buf379 = aie.buffer(%tile_3_2) {sym_name = "buf379"} : memref<64x64xbf16, 2 : i32> 
    %buf378 = aie.buffer(%tile_3_2) {sym_name = "buf378"} : memref<64x64xbf16, 2 : i32> 
    %buf377 = aie.buffer(%tile_3_2) {sym_name = "buf377"} : memref<64x64xbf16, 2 : i32> 
    %buf376 = aie.buffer(%tile_3_2) {sym_name = "buf376"} : memref<64x1xbf16, 2 : i32> 
    %buf375 = aie.buffer(%tile_3_2) {sym_name = "buf375"} : memref<64x1xbf16, 2 : i32> 
    %buf374 = aie.buffer(%tile_3_2) {sym_name = "buf374"} : memref<64x64xbf16, 2 : i32> 
    %buf373 = aie.buffer(%tile_3_2) {sym_name = "buf373"} : memref<64x64xbf16, 2 : i32> 
    %buf372 = aie.buffer(%tile_3_2) {sym_name = "buf372"} : memref<64x1xbf16, 2 : i32> 
    %buf371 = aie.buffer(%tile_3_2) {sym_name = "buf371"} : memref<64x1xbf16, 2 : i32> 
    %buf370 = aie.buffer(%tile_3_2) {sym_name = "buf370"} : memref<64x64xbf16, 2 : i32> 
    %buf369 = aie.buffer(%tile_3_2) {sym_name = "buf369"} : memref<64x1xbf16, 2 : i32> 
    %buf368 = aie.buffer(%tile_3_2) {sym_name = "buf368"} : memref<64x1xbf16, 2 : i32> 
    %buf367 = aie.buffer(%tile_3_2) {sym_name = "buf367"} : memref<64x1xbf16, 2 : i32> 
    %buf366 = aie.buffer(%tile_3_2) {sym_name = "buf366"} : memref<64x1xbf16, 2 : i32> 
    %buf365 = aie.buffer(%tile_3_2) {sym_name = "buf365"} : memref<64x1xbf16, 2 : i32> 
    %buf364 = aie.buffer(%tile_3_2) {sym_name = "buf364"} : memref<64x1xbf16, 2 : i32> 
    %buf363 = aie.buffer(%tile_2_2) {sym_name = "buf363"} : memref<64x1xbf16, 2 : i32> 
    %buf362 = aie.buffer(%tile_2_2) {sym_name = "buf362"} : memref<64x1xbf16, 2 : i32> 
    %buf361 = aie.buffer(%tile_2_2) {sym_name = "buf361"} : memref<64x64xbf16, 2 : i32> 
    %buf360 = aie.buffer(%tile_2_2) {sym_name = "buf360"} : memref<64x64xbf16, 2 : i32> 
    %buf359 = aie.buffer(%tile_2_2) {sym_name = "buf359"} : memref<64x64xbf16, 2 : i32> 
    %buf358 = aie.buffer(%tile_2_2) {sym_name = "buf358"} : memref<64x64xbf16, 2 : i32> 
    %buf357 = aie.buffer(%tile_2_2) {sym_name = "buf357"} : memref<64x64xbf16, 2 : i32> 
    %buf356 = aie.buffer(%tile_2_2) {sym_name = "buf356"} : memref<64x1xbf16, 2 : i32> 
    %buf355 = aie.buffer(%tile_2_2) {sym_name = "buf355"} : memref<64x1xbf16, 2 : i32> 
    %buf354 = aie.buffer(%tile_2_2) {sym_name = "buf354"} : memref<64x64xbf16, 2 : i32> 
    %buf353 = aie.buffer(%tile_2_2) {sym_name = "buf353"} : memref<64x64xbf16, 2 : i32> 
    %buf352 = aie.buffer(%tile_2_2) {sym_name = "buf352"} : memref<64x1xbf16, 2 : i32> 
    %buf351 = aie.buffer(%tile_2_2) {sym_name = "buf351"} : memref<64x1xbf16, 2 : i32> 
    %buf350 = aie.buffer(%tile_2_2) {sym_name = "buf350"} : memref<64x64xbf16, 2 : i32> 
    %buf349 = aie.buffer(%tile_2_2) {sym_name = "buf349"} : memref<64x1xbf16, 2 : i32> 
    %buf348 = aie.buffer(%tile_2_2) {sym_name = "buf348"} : memref<64x1xbf16, 2 : i32> 
    %buf347 = aie.buffer(%tile_2_2) {sym_name = "buf347"} : memref<64x1xbf16, 2 : i32> 
    %buf346 = aie.buffer(%tile_2_2) {sym_name = "buf346"} : memref<64x1xbf16, 2 : i32> 
    %buf345 = aie.buffer(%tile_2_2) {sym_name = "buf345"} : memref<64x1xbf16, 2 : i32> 
    %buf344 = aie.buffer(%tile_2_2) {sym_name = "buf344"} : memref<64x1xbf16, 2 : i32> 
    %buf343 = aie.buffer(%tile_1_2) {sym_name = "buf343"} : memref<64x1xbf16, 2 : i32> 
    %buf342 = aie.buffer(%tile_1_2) {sym_name = "buf342"} : memref<64x1xbf16, 2 : i32> 
    %buf341 = aie.buffer(%tile_1_2) {sym_name = "buf341"} : memref<64x64xbf16, 2 : i32> 
    %buf340 = aie.buffer(%tile_1_2) {sym_name = "buf340"} : memref<64x64xbf16, 2 : i32> 
    %buf339 = aie.buffer(%tile_1_2) {sym_name = "buf339"} : memref<64x64xbf16, 2 : i32> 
    %buf338 = aie.buffer(%tile_1_2) {sym_name = "buf338"} : memref<64x64xbf16, 2 : i32> 
    %buf337 = aie.buffer(%tile_1_2) {sym_name = "buf337"} : memref<64x64xbf16, 2 : i32> 
    %buf336 = aie.buffer(%tile_1_2) {sym_name = "buf336"} : memref<64x1xbf16, 2 : i32> 
    %buf335 = aie.buffer(%tile_1_2) {sym_name = "buf335"} : memref<64x1xbf16, 2 : i32> 
    %buf334 = aie.buffer(%tile_1_2) {sym_name = "buf334"} : memref<64x64xbf16, 2 : i32> 
    %buf333 = aie.buffer(%tile_1_2) {sym_name = "buf333"} : memref<64x64xbf16, 2 : i32> 
    %buf332 = aie.buffer(%tile_1_2) {sym_name = "buf332"} : memref<64x1xbf16, 2 : i32> 
    %buf331 = aie.buffer(%tile_1_2) {sym_name = "buf331"} : memref<64x1xbf16, 2 : i32> 
    %buf330 = aie.buffer(%tile_1_2) {sym_name = "buf330"} : memref<64x64xbf16, 2 : i32> 
    %buf329 = aie.buffer(%tile_1_2) {sym_name = "buf329"} : memref<64x1xbf16, 2 : i32> 
    %buf328 = aie.buffer(%tile_1_2) {sym_name = "buf328"} : memref<64x1xbf16, 2 : i32> 
    %buf327 = aie.buffer(%tile_1_2) {sym_name = "buf327"} : memref<64x1xbf16, 2 : i32> 
    %buf326 = aie.buffer(%tile_1_2) {sym_name = "buf326"} : memref<64x1xbf16, 2 : i32> 
    %buf325 = aie.buffer(%tile_1_2) {sym_name = "buf325"} : memref<64x1xbf16, 2 : i32> 
    %buf324 = aie.buffer(%tile_1_2) {sym_name = "buf324"} : memref<64x1xbf16, 2 : i32> 
    %buf323 = aie.buffer(%tile_0_2) {sym_name = "buf323"} : memref<64x1xbf16, 2 : i32> 
    %buf322 = aie.buffer(%tile_0_2) {sym_name = "buf322"} : memref<64x1xbf16, 2 : i32> 
    %buf321 = aie.buffer(%tile_0_2) {sym_name = "buf321"} : memref<64x64xbf16, 2 : i32> 
    %buf320 = aie.buffer(%tile_0_2) {sym_name = "buf320"} : memref<64x64xbf16, 2 : i32> 
    %buf319 = aie.buffer(%tile_0_2) {sym_name = "buf319"} : memref<64x64xbf16, 2 : i32> 
    %buf318 = aie.buffer(%tile_0_2) {sym_name = "buf318"} : memref<64x64xbf16, 2 : i32> 
    %buf317 = aie.buffer(%tile_0_2) {sym_name = "buf317"} : memref<64x64xbf16, 2 : i32> 
    %buf316 = aie.buffer(%tile_0_2) {sym_name = "buf316"} : memref<64x1xbf16, 2 : i32> 
    %buf315 = aie.buffer(%tile_0_2) {sym_name = "buf315"} : memref<64x1xbf16, 2 : i32> 
    %buf314 = aie.buffer(%tile_0_2) {sym_name = "buf314"} : memref<64x64xbf16, 2 : i32> 
    %buf313 = aie.buffer(%tile_0_2) {sym_name = "buf313"} : memref<64x64xbf16, 2 : i32> 
    %buf312 = aie.buffer(%tile_0_2) {sym_name = "buf312"} : memref<64x1xbf16, 2 : i32> 
    %buf311 = aie.buffer(%tile_0_2) {sym_name = "buf311"} : memref<64x1xbf16, 2 : i32> 
    %buf310 = aie.buffer(%tile_0_2) {sym_name = "buf310"} : memref<64x64xbf16, 2 : i32> 
    %buf309 = aie.buffer(%tile_0_2) {sym_name = "buf309"} : memref<64x1xbf16, 2 : i32> 
    %buf308 = aie.buffer(%tile_0_2) {sym_name = "buf308"} : memref<64x1xbf16, 2 : i32> 
    %buf307 = aie.buffer(%tile_0_2) {sym_name = "buf307"} : memref<64x1xbf16, 2 : i32> 
    %buf306 = aie.buffer(%tile_0_2) {sym_name = "buf306"} : memref<64x1xbf16, 2 : i32> 
    %buf305 = aie.buffer(%tile_0_2) {sym_name = "buf305"} : memref<64x1xbf16, 2 : i32> 
    %buf304 = aie.buffer(%tile_0_2) {sym_name = "buf304"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<2x512x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<2x512x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<2x512x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<2x512x64xbf16>
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_74, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf592 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_75, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf590 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_73, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf586 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf593) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf595) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf594) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf592, %buf591) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf589 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf589 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf591, %buf592, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf589 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf594, %buf588, %buf587) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf587, %buf593) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf589 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf590, %buf593) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf595, %buf587, %buf588) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf588, %buf595) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf585 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_74, Release, 1)
      aie.use_lock(%lock_3_5_75, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_73, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf585 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf591, %buf592, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf585 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf594, %buf584, %buf583) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf583, %buf593) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf585 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf586, %buf593) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf595, %buf583, %buf584) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf584, %buf595) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf593 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf594 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_85 = memref.collapse_shape %buf595 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_5_74, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_71, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf579 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf577 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_70, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf573 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf580) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf582) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf581) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf579, %buf578) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf576 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf576 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf578, %buf579, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf576 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf581, %buf575, %buf574) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf574, %buf580) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf576 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf577, %buf580) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf582, %buf574, %buf575) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf575, %buf582) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf572 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_71, Release, 1)
      aie.use_lock(%lock_2_5_72, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_70, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf572 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf578, %buf579, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf572 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf581, %buf571, %buf570) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf570, %buf580) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf572 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf573, %buf580) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf582, %buf570, %buf571) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf571, %buf582) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf580 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf581 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_85 = memref.collapse_shape %buf582 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_5_71, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_68, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf566 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_69, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf564 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_67, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf560 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf567) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf569) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf568) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf566, %buf565) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf563 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf563 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf565, %buf566, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf563 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf568, %buf562, %buf561) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf561, %buf567) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf563 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf564, %buf567) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf569, %buf561, %buf562) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf562, %buf569) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf559 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_68, Release, 1)
      aie.use_lock(%lock_1_5_69, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_67, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf559 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf565, %buf566, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf559 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf568, %buf558, %buf557) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf557, %buf567) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf559 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf560, %buf567) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf569, %buf557, %buf558) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf558, %buf569) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf567 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf568 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_85 = memref.collapse_shape %buf569 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_5_68, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_65, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf553 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_66, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf551 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_64, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf547 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf554) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf556) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf555) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf553, %buf552) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf550 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf550 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf552, %buf553, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf550 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf555, %buf549, %buf548) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf548, %buf554) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf550 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf551, %buf554) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf556, %buf548, %buf549) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf549, %buf556) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf546 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_65, Release, 1)
      aie.use_lock(%lock_0_5_66, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_64, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf546 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf552, %buf553, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf546 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf555, %buf545, %buf544) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf544, %buf554) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf546 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf547, %buf554) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf556, %buf544, %buf545) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf545, %buf556) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf554 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_84 = memref.collapse_shape %buf555 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_85 = memref.collapse_shape %buf556 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_5_65, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf540 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_63, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf538 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_61, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf534 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf541) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf543) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf542) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf540, %buf539) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf537 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf537 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf539, %buf540, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf537 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf542, %buf536, %buf535) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf535, %buf541) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf537 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf538, %buf541) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf543, %buf535, %buf536) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf536, %buf543) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf533 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_62, Release, 1)
      aie.use_lock(%lock_3_4_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_61, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf533 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf539, %buf540, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf533 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf542, %buf532, %buf531) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf531, %buf541) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf533 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf534, %buf541) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf543, %buf531, %buf532) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf532, %buf543) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf530 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf529 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf528 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf542, %buf527) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf529, %buf542) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf529, %buf542, %buf526) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf527, %buf542, %buf525) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf526, %buf530) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf525, %buf541) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf541, %buf530) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf524) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf528, %buf526, %buf524) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf543, %buf525, %buf524) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf524, %buf528) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf530 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf542 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf528 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_4_62, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_59, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf520 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf518 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_58, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf514 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf521) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf523) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf522) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf520, %buf519) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf517 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf517 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf519, %buf520, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf517 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf522, %buf516, %buf515) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf515, %buf521) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf517 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf518, %buf521) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf523, %buf515, %buf516) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf516, %buf523) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf513 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_59, Release, 1)
      aie.use_lock(%lock_2_4_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_58, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf513 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf519, %buf520, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf513 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf522, %buf512, %buf511) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf511, %buf521) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf513 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf514, %buf521) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf523, %buf511, %buf512) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf512, %buf523) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf510 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf509 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf508 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf522, %buf507) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf509, %buf522) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf509, %buf522, %buf506) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf507, %buf522, %buf505) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf506, %buf510) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf505, %buf521) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf521, %buf510) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf504) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf508, %buf506, %buf504) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf523, %buf505, %buf504) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf504, %buf508) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf510 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf522 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf508 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_4_59, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf500 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf498 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_55, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf494 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf501) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf503) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf502) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf500, %buf499) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf497 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf497 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf499, %buf500, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf497 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf502, %buf496, %buf495) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf495, %buf501) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf497 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf498, %buf501) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf503, %buf495, %buf496) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf496, %buf503) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf493 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_56, Release, 1)
      aie.use_lock(%lock_1_4_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_55, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf493 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf499, %buf500, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf493 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf502, %buf492, %buf491) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf491, %buf501) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf493 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf494, %buf501) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf503, %buf491, %buf492) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf492, %buf503) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf490 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf489 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf488 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf502, %buf487) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf489, %buf502) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf489, %buf502, %buf486) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf487, %buf502, %buf485) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf486, %buf490) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf485, %buf501) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf501, %buf490) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf484) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf488, %buf486, %buf484) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf503, %buf485, %buf484) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf484, %buf488) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf490 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf502 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf488 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_4_56, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf480 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf478 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_52, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf474 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf481) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf483) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf482) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf480, %buf479) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf477 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf477 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf479, %buf480, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf477 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf482, %buf476, %buf475) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf475, %buf481) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf477 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf478, %buf481) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf483, %buf475, %buf476) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf476, %buf483) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf473 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_53, Release, 1)
      aie.use_lock(%lock_0_4_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_52, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf473 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf479, %buf480, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf473 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf482, %buf472, %buf471) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf471, %buf481) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf473 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf474, %buf481) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf483, %buf471, %buf472) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf472, %buf483) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf470 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf469 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf468 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf482, %buf467) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf469, %buf482) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf469, %buf482, %buf466) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf467, %buf482, %buf465) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf466, %buf470) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf465, %buf481) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf481, %buf470) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf464) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf468, %buf466, %buf464) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf483, %buf465, %buf464) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf464, %buf468) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf470 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf482 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf468 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_4_53, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf460 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf458 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf454 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf461) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf463) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf462) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf460, %buf459) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf457 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf457 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf459, %buf460, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf457 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf462, %buf456, %buf455) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf455, %buf461) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf457 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf458, %buf461) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf463, %buf455, %buf456) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf456, %buf463) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf453 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_50, Release, 1)
      aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf453 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf459, %buf460, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf453 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf462, %buf452, %buf451) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf451, %buf461) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf453 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf454, %buf461) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf463, %buf451, %buf452) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf452, %buf463) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf450 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf449 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf448 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf462, %buf447) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf449, %buf462) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf449, %buf462, %buf446) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf447, %buf462, %buf445) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf446, %buf450) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf445, %buf461) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf461, %buf450) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf444) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf448, %buf446, %buf444) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf463, %buf445, %buf444) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf444, %buf448) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf450 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf462 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf448 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_3_50, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf440 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf438 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_46, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf434 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf441) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf443) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf442) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf440, %buf439) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf437 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf437 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf439, %buf440, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf437 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf442, %buf436, %buf435) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf435, %buf441) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf437 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf438, %buf441) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf443, %buf435, %buf436) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf436, %buf443) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf433 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf433 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf439, %buf440, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf433 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf442, %buf432, %buf431) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf431, %buf441) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf433 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf434, %buf441) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf443, %buf431, %buf432) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf432, %buf443) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf429 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf428 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf442, %buf427) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf429, %buf442) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf429, %buf442, %buf426) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf427, %buf442, %buf425) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf426, %buf430) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf425, %buf441) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf441, %buf430) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf424) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf428, %buf426, %buf424) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf443, %buf425, %buf424) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf424, %buf428) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf430 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf442 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf428 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_3_47, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_44, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf420 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf418 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf414 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf421) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf423) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf422) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf420, %buf419) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf417 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf417 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf419, %buf420, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf417 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf422, %buf416, %buf415) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf415, %buf421) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf417 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf418, %buf421) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf423, %buf415, %buf416) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf416, %buf423) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf413 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_44, Release, 1)
      aie.use_lock(%lock_1_3_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf413 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf419, %buf420, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf413 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf422, %buf412, %buf411) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf411, %buf421) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf413 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf414, %buf421) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf423, %buf411, %buf412) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf412, %buf423) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf410 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf409 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf408 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf422, %buf407) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf409, %buf422) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf409, %buf422, %buf406) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf407, %buf422, %buf405) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf406, %buf410) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf405, %buf421) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf421, %buf410) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf404) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf408, %buf406, %buf404) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf423, %buf405, %buf404) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf404, %buf408) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf410 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf422 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf408 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_3_44, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf400 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf398 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_40, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf394 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf401) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf403) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf402) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf400, %buf399) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf397 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf397 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf399, %buf400, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf397 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf402, %buf396, %buf395) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf395, %buf401) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf397 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf398, %buf401) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf403, %buf395, %buf396) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf396, %buf403) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf393 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_41, Release, 1)
      aie.use_lock(%lock_0_3_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_40, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf393 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf399, %buf400, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf393 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf402, %buf392, %buf391) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf391, %buf401) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf393 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf394, %buf401) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf403, %buf391, %buf392) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf392, %buf403) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf390 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf389 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf388 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf402, %buf387) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf389, %buf402) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf389, %buf402, %buf386) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf387, %buf402, %buf385) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf386, %buf390) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf385, %buf401) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf401, %buf390) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf384) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf388, %buf386, %buf384) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf403, %buf385, %buf384) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf384, %buf388) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      %collapse_shape_86 = memref.collapse_shape %buf390 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_86[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_87 = memref.collapse_shape %buf402 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_87[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_88 = memref.collapse_shape %buf388 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_88[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_3_41, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_39, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf370 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_36, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf380 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_37, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf378 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_35, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf374 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf381) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf383) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf382) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf380, %buf379) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf377 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf377 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf379, %buf380, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf377 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf382, %buf376, %buf375) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf375, %buf381) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf377 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf378, %buf381) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf383, %buf375, %buf376) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf376, %buf383) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_36, Release, 1)
      aie.use_lock(%lock_3_2_37, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf379, %buf380, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf382, %buf372, %buf371) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf371, %buf381) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf373 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf374, %buf381) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf383, %buf371, %buf372) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf372, %buf383) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf370 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf369 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf368 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf382, %buf367) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf369, %buf382) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf369, %buf382, %buf366) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf367, %buf382, %buf365) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf366, %buf370) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf365, %buf381) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf381, %buf370) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf364) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf368, %buf366, %buf364) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf383, %buf365, %buf364) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf364, %buf368) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf368, %buf370) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_39, Release, 1)
      aie.use_lock(%lock_3_2_36, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_34, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf350 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_31, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf360 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_32, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf358 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf354 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf361) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf363) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf362) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf360, %buf359) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf357 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf357 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf359, %buf360, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf357 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf362, %buf356, %buf355) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf355, %buf361) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf357 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf358, %buf361) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf363, %buf355, %buf356) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf356, %buf363) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf353 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_31, Release, 1)
      aie.use_lock(%lock_2_2_32, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf353 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf359, %buf360, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf353 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf362, %buf352, %buf351) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf351, %buf361) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf353 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf354, %buf361) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf363, %buf351, %buf352) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf352, %buf363) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf350 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf349 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf348 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf362, %buf347) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf349, %buf362) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf349, %buf362, %buf346) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf347, %buf362, %buf345) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf346, %buf350) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf345, %buf361) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf361, %buf350) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf344) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf348, %buf346, %buf344) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf363, %buf345, %buf344) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf344, %buf348) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf348, %buf350) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_34, Release, 1)
      aie.use_lock(%lock_2_2_31, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf330 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_28, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf340 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_27, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf338 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_25, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf334 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf341) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf343) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf342) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf340, %buf339) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf337 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf337 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf339, %buf340, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf337 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf342, %buf336, %buf335) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf335, %buf341) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf337 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf338, %buf341) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf343, %buf335, %buf336) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf336, %buf343) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_26, Release, 1)
      aie.use_lock(%lock_1_2_27, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf339, %buf340, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf342, %buf332, %buf331) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf331, %buf341) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf333 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf334, %buf341) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf343, %buf331, %buf332) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf332, %buf343) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf330 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf329 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf328 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf342, %buf327) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf329, %buf342) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf329, %buf342, %buf326) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf327, %buf342, %buf325) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf326, %buf330) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf325, %buf341) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf341, %buf330) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf324) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf328, %buf326, %buf324) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf343, %buf325, %buf324) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf324, %buf328) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf328, %buf330) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_29, Release, 1)
      aie.use_lock(%lock_1_2_26, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf310 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_23, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf320 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_22, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf318 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf314 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
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
      func.call @zero_fill_gp_bf16(%buf321) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf323) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf322) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf320, %buf319) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf317 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      %collapse_shape_76 = memref.collapse_shape %buf317 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf319, %buf320, %collapse_shape_76) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_77 = memref.collapse_shape %buf317 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_77, %buf322, %buf316, %buf315) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf315, %buf321) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_78 = memref.collapse_shape %buf317 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_78, %buf318, %buf321) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf323, %buf315, %buf316) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf316, %buf323) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2, Release, 1)
      %collapse_shape_79 = memref.collapse_shape %buf313 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_79) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_21, Release, 1)
      aie.use_lock(%lock_0_2_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      %collapse_shape_80 = memref.collapse_shape %buf313 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_a_b_bf16(%buf319, %buf320, %collapse_shape_80) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      %collapse_shape_81 = memref.collapse_shape %buf313 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @fused_softmax(%collapse_shape_81, %buf322, %buf312, %buf311) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf311, %buf321) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape_82 = memref.collapse_shape %buf313 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @matmul_g_b_bf16(%collapse_shape_82, %buf314, %buf321) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf323, %buf311, %buf312) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf312, %buf323) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2, Release, 1)
      %collapse_shape_83 = memref.collapse_shape %buf310 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_83[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_84 = memref.collapse_shape %buf309 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_84[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_85 = memref.collapse_shape %buf308 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_85[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      func.call @vector_copy_32elems(%c0_i32, %buf322, %buf307) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @maximum_up_u_bf16(%buf309, %buf322) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf309, %buf322, %buf306) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @exp_up_minus_u(%buf307, %buf322, %buf305) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf306, %buf310) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf305, %buf321) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @add_gp_g(%buf321, %buf310) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf304) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf308, %buf306, %buf304) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf323, %buf305, %buf304) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf304, %buf308) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @div_gp_sp(%buf308, %buf310) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_24, Release, 1)
      aie.use_lock(%lock_0_2_21, Release, 1)
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
      aie.use_lock(%lock_0_1_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf607 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_1_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf603 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_16, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf602 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_0_1_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf603 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_17, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf602 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_15, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1_18, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf607 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_19, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf606 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_13, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf601 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_11, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf600 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_1_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf601 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_12, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf600 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_10, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_1_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf606 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_14, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf605 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_8, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf599 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_6, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf598 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_2_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf599 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_7, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf598 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_2_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf605 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_9, Release, 1)
      aie.next_bd ^bb10
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf604 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb9
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf597 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf596 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_3_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf597 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf596 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb2)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_3_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf604 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_4, Release, 1)
      aie.next_bd ^bb10
    }
    aie.shim_dma_allocation @air_channel_0_1_0_4(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_5(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_6(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1_0_7(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_VIn_0_1_0_1(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_1_1_0_1(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_2_1_0_1(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_3_1_0_1(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @air_QK2L1_1_0_1_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_QK2L1_1_1_1_0(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_QK2L1_1_2_1_0(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_QK2L1_1_3_1_0(%shim_noc_tile_1_0, MM2S, 1)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 1 : i64, segment_unroll_y = 0 : i64}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 33 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 36 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 39 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 42 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 45 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 48 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 51 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 54 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg_0_0"}{
      airrt.herd_metadata {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 65 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 73 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 81 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 89 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 97 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 113 : i64, location = 0 : i64, row = 0 : i64}, {channel = 3 : i64, col = 0 : i64, id = 66 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 74 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 82 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 90 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 98 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 114 : i64, location = 0 : i64, row = 1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 67 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 75 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 83 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 91 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 99 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 115 : i64, location = 1 : i64, row = 2 : i64}, {channel = 3 : i64, col = 0 : i64, id = 68 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 76 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 84 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 92 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 100 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 116 : i64, location = 1 : i64, row = 3 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 33 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 36 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 39 : i64, location = 2 : i64, row = -1 : i64}, {channel = 3 : i64, col = 1 : i64, id = 42 : i64, location = 2 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 45 : i64, location = 3 : i64, row = -1 : i64}, {channel = 2 : i64, col = 2 : i64, id = 48 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 51 : i64, location = 3 : i64, row = -1 : i64}, {channel = 3 : i64, col = 3 : i64, id = 54 : i64, location = 3 : i64, row = -1 : i64}], sym_name = "attn_seg_1_0"}{
      airrt.herd_metadata {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 69 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 77 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 85 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 93 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 101 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 117 : i64, location = 0 : i64, row = 0 : i64}, {channel = 3 : i64, col = 0 : i64, id = 70 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 78 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 86 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 94 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 102 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 118 : i64, location = 0 : i64, row = 1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 71 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 79 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 87 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 95 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 103 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 119 : i64, location = 1 : i64, row = 2 : i64}, {channel = 3 : i64, col = 0 : i64, id = 72 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 80 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 88 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 96 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 104 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 120 : i64, location = 1 : i64, row = 3 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
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
  func.func @attention_bf16(%arg0: memref<2x512x64xbf16>, %arg1: memref<2x512x64xbf16>, %arg2: memref<2x512x64xbf16>, %arg3: memref<2x512x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16>, memref<2x512x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c2_0 = arith.constant 2 : index
      %c1_1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg5, %arg4]
      %2 = air.channel.put async  @QK2L1_0_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 1 : i32, metadataArray = [{base = "air_QK2L1_0_0_0_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %3 = air.channel.put async  @QK2L1_0_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 2 : i32, metadataArray = [{base = "air_QK2L1_0_1_0_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %4 = air.channel.put async  @QK2L1_0_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 3 : i32, metadataArray = [{base = "air_QK2L1_0_2_0_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %5 = air.channel.put async  @QK2L1_0_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 4 : i32, metadataArray = [{base = "air_QK2L1_0_3_0_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %6 = affine.apply #map1()[%arg5]
      %7 = air.channel.put async  @QK2L1_0_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %6] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 5 : i32, metadataArray = [{base = "air_QK2L1_0_0_0_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %8 = affine.apply #map2()[%arg5]
      %9 = air.channel.put async  @QK2L1_0_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %8] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 6 : i32, metadataArray = [{base = "air_QK2L1_0_1_0_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %10 = affine.apply #map3()[%arg5]
      %11 = air.channel.put async  @QK2L1_0_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %10] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 7 : i32, metadataArray = [{base = "air_QK2L1_0_2_0_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %12 = affine.apply #map4()[%arg5]
      %13 = air.channel.put async  @QK2L1_0_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %12] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 8 : i32, metadataArray = [{base = "air_QK2L1_0_3_0_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %14 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %6] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 9 : i32, metadataArray = [{base = "air_VIn_0_0_0", index = 0 : i32}, {base = "air_VIn_0_1_0_1", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %15 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %8] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 10 : i32, metadataArray = [{base = "air_VIn_1_0_0", index = 0 : i32}, {base = "air_VIn_1_1_0_1", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %16 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %10] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 11 : i32, metadataArray = [{base = "air_VIn_2_0_0", index = 0 : i32}, {base = "air_VIn_2_1_0_1", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %17 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %12] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 12 : i32, metadataArray = [{base = "air_VIn_3_0_0", index = 0 : i32}, {base = "air_VIn_3_1_0_1", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %18 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %1] [%c64, %c64] [%c64, %c1_1]) {id = 13 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_4", index = 4 : i32}, {base = "air_channel_0_1_0_5", index = 5 : i32}, {base = "air_channel_0_1_0_6", index = 6 : i32}, {base = "air_channel_0_1_0_7", index = 7 : i32}]} : (memref<2x512x64xbf16>)
      %19 = air.channel.get async  @channel_0[%c1_1, %c0] (%arg11[%c64, %1] [%c64, %c64] [%c64, %c1_1]) {id = 14 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_4", index = 4 : i32}, {base = "air_channel_0_1_0_5", index = 5 : i32}, {base = "air_channel_0_1_0_6", index = 6 : i32}, {base = "air_channel_0_1_0_7", index = 7 : i32}]} : (memref<2x512x64xbf16>)
      %20 = air.channel.get async  @channel_0[%c2_0, %c0] (%arg11[%c128, %1] [%c64, %c64] [%c64, %c1_1]) {id = 15 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_4", index = 4 : i32}, {base = "air_channel_0_1_0_5", index = 5 : i32}, {base = "air_channel_0_1_0_6", index = 6 : i32}, {base = "air_channel_0_1_0_7", index = 7 : i32}]} : (memref<2x512x64xbf16>)
      %21 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %1] [%c64, %c64] [%c64, %c1_1]) {id = 16 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_4", index = 4 : i32}, {base = "air_channel_0_1_0_5", index = 5 : i32}, {base = "air_channel_0_1_0_6", index = 6 : i32}, {base = "air_channel_0_1_0_7", index = 7 : i32}]} : (memref<2x512x64xbf16>)
      %22 = affine.apply #map5()[%arg5, %arg4]
      %23 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 17 : i32, metadataArray = [{base = "air_QK2L1_1_0_1_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %24 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 18 : i32, metadataArray = [{base = "air_QK2L1_1_1_1_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %25 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 19 : i32, metadataArray = [{base = "air_QK2L1_1_2_1_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %26 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %22] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 20 : i32, metadataArray = [{base = "air_QK2L1_1_3_1_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %27 = affine.apply #map6()[%arg5]
      %28 = air.channel.put async  @QK2L1_1_0[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %27] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 21 : i32, metadataArray = [{base = "air_QK2L1_1_0_1_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %29 = affine.apply #map7()[%arg5]
      %30 = air.channel.put async  @QK2L1_1_1[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %29] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 22 : i32, metadataArray = [{base = "air_QK2L1_1_1_1_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %31 = affine.apply #map8()[%arg5]
      %32 = air.channel.put async  @QK2L1_1_2[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %31] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 23 : i32, metadataArray = [{base = "air_QK2L1_1_2_1_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %33 = affine.apply #map9()[%arg5]
      %34 = air.channel.put async  @QK2L1_1_3[%c0, %c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %33] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 24 : i32, metadataArray = [{base = "air_QK2L1_1_3_1_0", index = 0 : i32}]} : (memref<2x512x64xbf16>)
      %35 = air.channel.put async  @VIn_0[%c1_1] (%arg10[%c0, %c0, %27] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 25 : i32, metadataArray = [{base = "air_VIn_0_0_0", index = 0 : i32}, {base = "air_VIn_0_1_0_1", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %36 = air.channel.put async  @VIn_1[%c1_1] (%arg10[%c0, %c0, %29] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 26 : i32, metadataArray = [{base = "air_VIn_1_0_0", index = 0 : i32}, {base = "air_VIn_1_1_0_1", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %37 = air.channel.put async  @VIn_2[%c1_1] (%arg10[%c0, %c0, %31] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 27 : i32, metadataArray = [{base = "air_VIn_2_0_0", index = 0 : i32}, {base = "air_VIn_2_1_0_1", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %38 = air.channel.put async  @VIn_3[%c1_1] (%arg10[%c0, %c0, %33] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 28 : i32, metadataArray = [{base = "air_VIn_3_0_0", index = 0 : i32}, {base = "air_VIn_3_1_0_1", index = 1 : i32}]} : (memref<2x512x64xbf16>)
      %39 = air.channel.get async  @channel_0[%c0, %c1_1] (%arg11[%c0, %22] [%c64, %c64] [%c64, %c1_1]) {id = 29 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_4", index = 4 : i32}, {base = "air_channel_0_1_0_5", index = 5 : i32}, {base = "air_channel_0_1_0_6", index = 6 : i32}, {base = "air_channel_0_1_0_7", index = 7 : i32}]} : (memref<2x512x64xbf16>)
      %40 = air.channel.get async  @channel_0[%c1_1, %c1_1] (%arg11[%c64, %22] [%c64, %c64] [%c64, %c1_1]) {id = 30 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_4", index = 4 : i32}, {base = "air_channel_0_1_0_5", index = 5 : i32}, {base = "air_channel_0_1_0_6", index = 6 : i32}, {base = "air_channel_0_1_0_7", index = 7 : i32}]} : (memref<2x512x64xbf16>)
      %41 = air.channel.get async  @channel_0[%c2_0, %c1_1] (%arg11[%c128, %22] [%c64, %c64] [%c64, %c1_1]) {id = 31 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_4", index = 4 : i32}, {base = "air_channel_0_1_0_5", index = 5 : i32}, {base = "air_channel_0_1_0_6", index = 6 : i32}, {base = "air_channel_0_1_0_7", index = 7 : i32}]} : (memref<2x512x64xbf16>)
      %42 = air.channel.get async  @channel_0[%c3, %c1_1] (%arg11[%c192, %22] [%c64, %c64] [%c64, %c1_1]) {id = 32 : i32, metadataArray = [{base = "air_channel_0_0_0_0", index = 0 : i32}, {base = "air_channel_0_0_0_1", index = 1 : i32}, {base = "air_channel_0_0_0_2", index = 2 : i32}, {base = "air_channel_0_0_0_3", index = 3 : i32}, {base = "air_channel_0_1_0_4", index = 4 : i32}, {base = "air_channel_0_1_0_5", index = 5 : i32}, {base = "air_channel_0_1_0_6", index = 6 : i32}, {base = "air_channel_0_1_0_7", index = 7 : i32}]} : (memref<2x512x64xbf16>)
      %43 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2_0, %arg15=%c1_1) attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c3_2 = arith.constant 3 : index
        %c64_3 = arith.constant 64 : index
        %c8_4 = arith.constant 8 : index
        %c1_5 = arith.constant 1 : index
        %c2_6 = arith.constant 2 : index
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_9, %results_10 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_11, %results_12 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_15, %results_16 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %44 = air.channel.get async [%async_token_15]  @VIn_0[%arg12] (%results_16[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
        %45 = arith.cmpi eq, %arg12, %c0_7 : index
        %46 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%44]  @V2L1_0_0[%c0_7, %c0_7, %c0_7] (%results_16[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%44]  @V2L1_0_1[%c0_7, %c0_7, %c0_7] (%results_16[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_17 = air.execute [%46, %44] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %47 = air.channel.get async [%async_token_18]  @VIn_0[%arg12] (%results_19[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
        %48 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%47]  @V2L1_0_0[%c0_7, %c0_7, %c0_7] (%results_19[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%47]  @V2L1_0_1[%c0_7, %c0_7, %c0_7] (%results_19[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_20 = air.execute [%48, %47] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %49 = air.channel.get async [%async_token_21]  @VIn_1[%arg12] (%results_22[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
        %50 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%49]  @V2L1_1_0[%c0_7, %c0_7, %c0_7] (%results_22[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%49]  @V2L1_1_1[%c0_7, %c0_7, %c0_7] (%results_22[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_23 = air.execute [%50, %49] {
          memref.dealloc %results_22 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %51 = air.channel.get async [%async_token_24]  @VIn_1[%arg12] (%results_25[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
        %52 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%51]  @V2L1_1_0[%c0_7, %c0_7, %c0_7] (%results_25[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 43 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%51]  @V2L1_1_1[%c0_7, %c0_7, %c0_7] (%results_25[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 44 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_26 = air.execute [%52, %51] {
          memref.dealloc %results_25 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %53 = air.channel.get async [%async_token_27]  @VIn_2[%arg12] (%results_28[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 1 : i32>)
        %54 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%53]  @V2L1_2_0[%c0_7, %c0_7, %c0_7] (%results_28[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 46 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%53]  @V2L1_2_1[%c0_7, %c0_7, %c0_7] (%results_28[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 47 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_29 = air.execute [%54, %53] {
          memref.dealloc %results_28 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_30, %results_31 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %55 = air.channel.get async [%async_token_30]  @VIn_2[%arg12] (%results_31[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 1 : i32>)
        %56 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%55]  @V2L1_2_0[%c0_7, %c0_7, %c0_7] (%results_31[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 49 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%55]  @V2L1_2_1[%c0_7, %c0_7, %c0_7] (%results_31[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 50 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_32 = air.execute [%56, %55] {
          memref.dealloc %results_31 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_33, %results_34 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %57 = air.channel.get async [%async_token_33]  @VIn_3[%arg12] (%results_34[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 1 : i32>)
        %58 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%57]  @V2L1_3_0[%c0_7, %c0_7, %c0_7] (%results_34[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 52 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%57]  @V2L1_3_1[%c0_7, %c0_7, %c0_7] (%results_34[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 53 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_35 = air.execute [%58, %57] {
          memref.dealloc %results_34 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_36, %results_37 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %59 = air.channel.get async [%async_token_36]  @VIn_3[%arg12] (%results_37[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 1 : i32>)
        %60 = scf.if %45 -> (!air.async.token) {
          %70 = air.channel.put async [%59]  @V2L1_3_0[%c0_7, %c0_7, %c0_7] (%results_37[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 55 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        } else {
          %70 = air.channel.put async [%59]  @V2L1_3_1[%c0_7, %c0_7, %c0_7] (%results_37[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 56 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %70 : !air.async.token
        }
        %async_token_38 = air.execute [%60, %59] {
          memref.dealloc %results_37 : memref<64x64xbf16, 1 : i32>
        }
        %61 = air.channel.get async [%async_token]  @Gp2L2[%c0_7, %c0_7] (%results[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 1 : i32>)
        %62 = air.channel.get async [%async_token_9]  @Gp2L2[%c1_5, %c0_7] (%results_10[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 1 : i32>)
        %63 = air.channel.get async [%async_token_11]  @Gp2L2[%c2_6, %c0_7] (%results_12[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 1 : i32>)
        %64 = air.channel.get async [%async_token_13]  @Gp2L2[%c3_2, %c0_7] (%results_14[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 1 : i32>)
        %65 = air.channel.put async [%61]  @channel_0[%c0_7, %arg12] (%results[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 1 : i32>)
        %66 = air.channel.put async [%62]  @channel_0[%c1_5, %arg12] (%results_10[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 1 : i32>)
        %67 = air.channel.put async [%63]  @channel_0[%c2_6, %arg12] (%results_12[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 1 : i32>)
        %68 = air.channel.put async [%64]  @channel_0[%c3_2, %arg12] (%results_14[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 1 : i32>)
        %69 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c64_43 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c0_44 = arith.constant 0 : index
          %c1_45 = arith.constant 1 : index
          %c8_46 = arith.constant 8 : index
          %c512_47 = arith.constant 512 : index
          %async_token_48, %results_49 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_50, %results_51 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
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
          %async_token_58 = air.execute [%async_token_52] {
            func.call @zero_fill_gp_bf16(%results_53) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_59 = air.execute [%async_token_48] {
            func.call @zero_fill_sp_bf16(%results_49) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_60 = air.execute [%async_token_50] {
            func.call @neg_inf_fill_up_bf16(%results_51) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %70 = arith.cmpi eq, %arg20, %c0_44 : index
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %71 = arith.index_cast %arg16 : index to i32
          %72 = arith.cmpi eq, %71, %c0_i32 : i32
          scf.if %72 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %73 = arith.cmpi eq, %71, %c1_i32 : i32
          scf.if %73 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %74 = arith.cmpi eq, %71, %c2_i32 : i32
          scf.if %74 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 92 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 93 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 94 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 95 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 96 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %75 = arith.cmpi eq, %71, %c3_i32 : i32
          scf.if %75 {
            %async_token_104 = air.execute [%async_token_54, %async_token_56] {
              func.call @copy_tile(%results_55, %results_57) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %async_token_61, %results_62 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_63, %results_64 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_65 = air.execute [%async_token_63] {
            %collapse_shape = memref.collapse_shape %results_64 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 97 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 98 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 99 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 100 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 101 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 102 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 103 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 104 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %76 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_61]  @V2L1_0_0[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 105 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_61]  @V2L1_0_1[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 106 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %77 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_61, %76]  @V2L1_1_0[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 107 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_61, %76]  @V2L1_1_1[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 108 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %78 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_61, %77]  @V2L1_2_0[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 109 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_61, %77]  @V2L1_2_1[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 110 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %79 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_61, %78]  @V2L1_3_0[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 111 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_61, %78]  @V2L1_3_1[%c0_44, %arg17, %arg16] (%results_62[] [] []) {id = 112 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %async_token_66 = air.execute [%async_token_65, %async_token_56, %async_token_54] {
            %collapse_shape = memref.collapse_shape %results_64 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_57, %results_55, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_69, %results_70 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_71 = air.execute [%async_token_50, %async_token_63, %async_token_69, %async_token_67, %async_token_66] {
            %collapse_shape = memref.collapse_shape %results_64 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_51, %results_68, %results_70) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_72 = air.execute [%async_token_52, %async_token_71] {
            func.call @mul_r_gp(%results_70, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_73 = air.execute [%79, %async_token_72, %async_token_61, %async_token_63] {
            %collapse_shape = memref.collapse_shape %results_64 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_62, %results_53) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_74 = air.execute [%async_token_48, %async_token_72] {
            func.call @accum_sp_r_s(%results_49, %results_70, %results_68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_75 = air.execute [%async_token_74] {
            func.call @vector_copy_32elems(%c0_i32, %results_68, %results_49) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_76 = air.execute [%async_token_75] {
            memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_77 = air.execute [%async_token_74] {
            memref.dealloc %results_70 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_78 = air.execute [%async_token_71, %async_token_73] {
            memref.dealloc %results_64 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_79 = air.execute [%76, %77, %78, %async_token_73] {
            memref.dealloc %results_62 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_80, %results_81 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_82, %results_83 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_84 = air.execute [%async_token_82] {
            %collapse_shape = memref.collapse_shape %results_83 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
          }
          scf.if %70 {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_0_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 113 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_0_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 114 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 115 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_0_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 116 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          } else {
            %86 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %87 = air.channel.get async [%async_token_54]  @QK2L1_1_0[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 117 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %87 : !air.async.token
            } else {
              %87 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %88 = air.channel.get async [%async_token_54]  @QK2L1_1_1[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 118 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %88 : !air.async.token
              } else {
                %88 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_2[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 119 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                } else {
                  %89 = air.channel.get async [%async_token_54]  @QK2L1_1_3[%c0_44, %c0_44, %arg16] (%results_55[] [] []) {id = 120 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %89 : !air.async.token
                }
                affine.yield %88 : !air.async.token
              }
              affine.yield %87 : !air.async.token
            }
          }
          %80 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_80]  @V2L1_0_0[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 121 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_80]  @V2L1_0_1[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 122 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %81 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%77, %async_token_80, %80]  @V2L1_1_0[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 123 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%77, %async_token_80, %80]  @V2L1_1_1[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 124 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %82 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%78, %async_token_80, %81]  @V2L1_2_0[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 125 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%78, %async_token_80, %81]  @V2L1_2_1[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 126 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %83 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %86 = scf.if %70 -> (!air.async.token) {
              %87 = air.channel.get async [%async_token_80, %82]  @V2L1_3_0[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 127 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            } else {
              %87 = air.channel.get async [%async_token_80, %82]  @V2L1_3_1[%c0_44, %arg17, %arg16] (%results_81[] [] []) {id = 128 : i32} : (memref<64x64xbf16, 2 : i32>)
              scf.yield %87 : !air.async.token
            }
            affine.yield %86 : !air.async.token
          } else {
            %86 = air.wait_all async 
            affine.yield %86 : !air.async.token
          }
          %async_token_85 = air.execute [%async_token_84, %async_token_56, %async_token_54] {
            %collapse_shape = memref.collapse_shape %results_83 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_a_b_bf16(%results_57, %results_55, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
          }
          %async_token_86, %results_87 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_88, %results_89 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_90 = air.execute [%async_token_50, %async_token_82, %async_token_88, %async_token_86, %async_token_85] {
            %collapse_shape = memref.collapse_shape %results_83 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %results_51, %results_87, %results_89) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_91 = air.execute [%async_token_52, %async_token_90] {
            func.call @mul_r_gp(%results_89, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_92 = air.execute [%83, %async_token_91, %async_token_80, %async_token_82] {
            %collapse_shape = memref.collapse_shape %results_83 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @matmul_g_b_bf16(%collapse_shape, %results_81, %results_53) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_93 = air.execute [%async_token_48, %async_token_91] {
            func.call @accum_sp_r_s(%results_49, %results_89, %results_87) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_94 = air.execute [%async_token_93] {
            func.call @vector_copy_32elems(%c0_i32, %results_87, %results_49) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_95 = air.execute [%async_token_94] {
            memref.dealloc %results_87 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_96 = air.execute [%async_token_93] {
            memref.dealloc %results_89 : memref<64x1xbf16, 2 : i32>
          }
          %84 = air.wait_all async [%async_token_92, %async_token_94] 
          %async_token_97 = air.execute [%async_token_90, %async_token_92] {
            memref.dealloc %results_83 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_98 = air.execute [%80, %81, %82, %async_token_92] {
            memref.dealloc %results_81 : memref<64x64xbf16, 2 : i32>
          }
          %85 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %86 = arith.subi %arg17, %c1_45 : index
            %87 = air.channel.put async [%async_token_52, %84]  @cascade_gp[%arg16, %86] (%results_53[] [] []) {id = 129 : i32} : (memref<64x64xbf16, 2 : i32>)
            %88 = air.channel.put async [%async_token_50]  @cascade_up[%arg16, %86] (%results_51[] [] []) {id = 130 : i32} : (memref<64x1xbf16, 2 : i32>)
            %89 = air.channel.put async [%async_token_48, %84]  @cascade_sp[%arg16, %86] (%results_49[] [] []) {id = 131 : i32} : (memref<64x1xbf16, 2 : i32>)
            %90 = air.wait_all async [%87, %88, %89] 
            affine.yield %90 : !air.async.token
          } else {
            %86 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_104, %results_105 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_106, %results_107 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_108, %results_109 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %87 = air.channel.get async [%async_token_104]  @cascade_gp[%arg16, %arg17] (%results_105[] [] []) {id = 132 : i32} : (memref<64x64xbf16, 2 : i32>)
              %88 = air.channel.get async [%async_token_106]  @cascade_up[%arg16, %arg17] (%results_107[] [] []) {id = 133 : i32} : (memref<64x1xbf16, 2 : i32>)
              %89 = air.channel.get async [%async_token_108]  @cascade_sp[%arg16, %arg17] (%results_109[] [] []) {id = 134 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_110, %results_111 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_110, %async_token_50] {
                func.call @vector_copy_32elems(%c0_i32, %results_51, %results_111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_113 = air.execute [%88, %async_token_112] {
                func.call @maximum_up_u_bf16(%results_107, %results_51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_114, %results_115 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_116 = air.execute [%async_token_50, %async_token_114, %async_token_113] {
                func.call @exp_up_minus_u(%results_107, %results_51, %results_115) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_117, %results_118 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_119 = air.execute [%async_token_116, %async_token_117] {
                func.call @exp_up_minus_u(%results_111, %results_51, %results_118) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_120 = air.execute [%async_token_116, %87] {
                func.call @mul_r_gp(%results_115, %results_105) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_121 = air.execute [%async_token_52, %async_token_119] {
                func.call @mul_r_gp(%results_118, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_122 = air.execute [%async_token_120, %async_token_121] {
                func.call @add_gp_g(%results_53, %results_105) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_123, %results_124 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_125 = air.execute [%async_token_123] {
                func.call @zero_fill_sp_bf16(%results_124) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_126 = air.execute [%async_token_125, %async_token_120, %89] {
                func.call @accum_sp_r_s(%results_109, %results_115, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_48, %async_token_126, %async_token_121] {
                func.call @accum_sp_r_s(%results_49, %results_118, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_128 = air.execute [%async_token_127] {
                func.call @vector_copy_32elems(%c0_i32, %results_124, %results_109) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %90 = arith.subi %arg17, %c1_45 : index
              %91 = air.channel.put async [%async_token_122]  @cascade_gp[%arg16, %90] (%results_105[] [] []) {id = 135 : i32} : (memref<64x64xbf16, 2 : i32>)
              %92 = air.channel.put async [%async_token_50, %async_token_119]  @cascade_up[%arg16, %90] (%results_51[] [] []) {id = 136 : i32} : (memref<64x1xbf16, 2 : i32>)
              %93 = air.channel.put async [%async_token_128]  @cascade_sp[%arg16, %90] (%results_109[] [] []) {id = 137 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_129 = air.execute [%91] {
                memref.dealloc %results_105 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_130 = air.execute [%async_token_116] {
                memref.dealloc %results_107 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_131 = air.execute [%93] {
                memref.dealloc %results_109 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_132 = air.execute [%async_token_119] {
                memref.dealloc %results_111 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_133 = air.execute [%async_token_126] {
                memref.dealloc %results_115 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_134 = air.execute [%async_token_127] {
                memref.dealloc %results_118 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_128] {
                memref.dealloc %results_124 : memref<64x1xbf16, 2 : i32>
              }
              %94 = air.wait_all async [%91, %92, %93] 
              affine.yield %94 : !air.async.token
            } else {
              %async_token_104, %results_105 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_106, %results_107 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_108, %results_109 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %87 = air.channel.get async [%async_token_104]  @cascade_gp[%arg16, %arg17] (%results_105[] [] []) {id = 138 : i32} : (memref<64x64xbf16, 2 : i32>)
              %88 = air.channel.get async [%async_token_106]  @cascade_up[%arg16, %arg17] (%results_107[] [] []) {id = 139 : i32} : (memref<64x1xbf16, 2 : i32>)
              %89 = air.channel.get async [%async_token_108]  @cascade_sp[%arg16, %arg17] (%results_109[] [] []) {id = 140 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_110, %results_111 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_112 = air.execute [%async_token_110, %async_token_50] {
                func.call @vector_copy_32elems(%c0_i32, %results_51, %results_111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_113 = air.execute [%88, %async_token_112] {
                func.call @maximum_up_u_bf16(%results_107, %results_51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_114, %results_115 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_116 = air.execute [%async_token_50, %async_token_114, %async_token_113] {
                func.call @exp_up_minus_u(%results_107, %results_51, %results_115) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_117, %results_118 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_119 = air.execute [%async_token_116, %async_token_117] {
                func.call @exp_up_minus_u(%results_111, %results_51, %results_118) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_120 = air.execute [%async_token_116, %87] {
                func.call @mul_r_gp(%results_115, %results_105) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_121 = air.execute [%async_token_52, %async_token_119] {
                func.call @mul_r_gp(%results_118, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_122 = air.execute [%async_token_120, %async_token_121] {
                func.call @add_gp_g(%results_53, %results_105) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_123, %results_124 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_125 = air.execute [%async_token_123] {
                func.call @zero_fill_sp_bf16(%results_124) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_126 = air.execute [%async_token_125, %async_token_120, %89] {
                func.call @accum_sp_r_s(%results_109, %results_115, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_127 = air.execute [%async_token_48, %async_token_126, %async_token_121] {
                func.call @accum_sp_r_s(%results_49, %results_118, %results_124) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_128 = air.execute [%async_token_127] {
                func.call @vector_copy_32elems(%c0_i32, %results_124, %results_109) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_129 = air.execute [%async_token_128, %async_token_122] {
                func.call @div_gp_sp(%results_109, %results_105) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %90 = air.channel.put async [%async_token_129]  @Gp2L2[%arg16, %c0_44] (%results_105[%c0_44, %c0_44, %c0_44] [%c64_43, %c8_46, %c8_46] [%c8_46, %c512_47, %c1_45]) {id = 141 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_130 = air.execute [%90] {
                memref.dealloc %results_105 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_131 = air.execute [%async_token_116] {
                memref.dealloc %results_107 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_132 = air.execute [%async_token_129] {
                memref.dealloc %results_109 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_133 = air.execute [%async_token_119] {
                memref.dealloc %results_111 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_134 = air.execute [%async_token_126] {
                memref.dealloc %results_115 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_135 = air.execute [%async_token_127] {
                memref.dealloc %results_118 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_136 = air.execute [%async_token_128] {
                memref.dealloc %results_124 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %90 : !air.async.token
            }
            affine.yield %84 : !air.async.token
          }
          %async_token_99 = air.execute {
            memref.dealloc %results_57 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_100 = air.execute {
            memref.dealloc %results_55 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_101 = air.execute [%85, %84, %async_token_58] {
            memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_102 = air.execute [%async_token_60, %85] {
            memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_103 = air.execute [%85, %84, %async_token_59] {
            memref.dealloc %results_49 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_39 = air.execute [%68] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_40 = air.execute [%67] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_41 = air.execute [%66] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_42 = air.execute [%65] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%async_token_17, %async_token_20, %async_token_23, %async_token_26, %async_token_29, %async_token_32, %async_token_35, %async_token_38, %69, %async_token_39, %async_token_40, %async_token_41, %async_token_42]  {air.segment_end}
      }
    }
    return
  }
}
