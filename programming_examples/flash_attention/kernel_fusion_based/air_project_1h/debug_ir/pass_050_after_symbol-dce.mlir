module {
  aie.device(npu2) @attn_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
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
    %lock_7_1 = aie.lock(%mem_tile_7_1, 3) {init = 1 : i32}
    %lock_7_1_0 = aie.lock(%mem_tile_7_1, 2) {init = 0 : i32}
    %lock_7_1_1 = aie.lock(%mem_tile_7_1, 1) {init = 1 : i32}
    %lock_7_1_2 = aie.lock(%mem_tile_7_1, 0) {init = 0 : i32}
    %lock_6_1 = aie.lock(%mem_tile_6_1, 3) {init = 1 : i32}
    %lock_6_1_3 = aie.lock(%mem_tile_6_1, 2) {init = 0 : i32}
    %lock_6_1_4 = aie.lock(%mem_tile_6_1, 1) {init = 1 : i32}
    %lock_6_1_5 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_5_1 = aie.lock(%mem_tile_5_1, 3) {init = 1 : i32}
    %lock_5_1_6 = aie.lock(%mem_tile_5_1, 2) {init = 0 : i32}
    %lock_5_1_7 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %lock_5_1_8 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_4_1 = aie.lock(%mem_tile_4_1, 3) {init = 1 : i32}
    %lock_4_1_9 = aie.lock(%mem_tile_4_1, 2) {init = 0 : i32}
    %lock_4_1_10 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %lock_4_1_11 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_3_1 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_12 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_13 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_14 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_15 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 2 : i32}
    %lock_0_2_16 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_17 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_18 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_19 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_20 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 2 : i32}
    %lock_1_2_21 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_22 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_23 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_24 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_25 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 2 : i32}
    %lock_2_2_26 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_27 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_28 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_29 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_30 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 2 : i32}
    %lock_3_2_31 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_32 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_33 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_34 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_35 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 2 : i32}
    %lock_0_3_36 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_37 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_38 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 3) {init = 2 : i32}
    %lock_1_3_39 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_40 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_41 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 3) {init = 2 : i32}
    %lock_2_3_42 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_43 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_44 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 3) {init = 2 : i32}
    %lock_3_3_45 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_46 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_47 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 2 : i32}
    %lock_0_4_48 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_49 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_50 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 3) {init = 2 : i32}
    %lock_1_4_51 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_52 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_53 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 3) {init = 2 : i32}
    %lock_2_4_54 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_55 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_56 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 3) {init = 2 : i32}
    %lock_3_4_57 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_58 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_59 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 2 : i32}
    %lock_0_5_60 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_61 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_62 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 3) {init = 2 : i32}
    %lock_1_5_63 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_64 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_65 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 3) {init = 2 : i32}
    %lock_2_5_66 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_67 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_68 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 3) {init = 2 : i32}
    %lock_3_5_69 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_70 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_71 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf303 = aie.buffer(%mem_tile_0_1) {sym_name = "buf303"} : memref<64x64xbf16, 1 : i32> 
    %buf302 = aie.buffer(%mem_tile_1_1) {sym_name = "buf302"} : memref<64x64xbf16, 1 : i32> 
    %buf301 = aie.buffer(%mem_tile_2_1) {sym_name = "buf301"} : memref<64x64xbf16, 1 : i32> 
    %buf300 = aie.buffer(%mem_tile_3_1) {sym_name = "buf300"} : memref<64x64xbf16, 1 : i32> 
    %buf299 = aie.buffer(%mem_tile_4_1) {sym_name = "buf299"} : memref<64x64xbf16, 1 : i32> 
    %buf298 = aie.buffer(%mem_tile_4_1) {sym_name = "buf298"} : memref<64x64xbf16, 1 : i32> 
    %buf297 = aie.buffer(%mem_tile_5_1) {sym_name = "buf297"} : memref<64x64xbf16, 1 : i32> 
    %buf296 = aie.buffer(%mem_tile_5_1) {sym_name = "buf296"} : memref<64x64xbf16, 1 : i32> 
    %buf295 = aie.buffer(%mem_tile_6_1) {sym_name = "buf295"} : memref<64x64xbf16, 1 : i32> 
    %buf294 = aie.buffer(%mem_tile_6_1) {sym_name = "buf294"} : memref<64x64xbf16, 1 : i32> 
    %buf293 = aie.buffer(%mem_tile_7_1) {sym_name = "buf293"} : memref<64x64xbf16, 1 : i32> 
    %buf292 = aie.buffer(%mem_tile_7_1) {sym_name = "buf292"} : memref<64x64xbf16, 1 : i32> 
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
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<256x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<512x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<512x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<256x64xbf16>
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_70, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf288 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_71, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf286 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_69, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf282 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_69, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_5 = aie.core(%tile_3_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf289) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf291) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf290) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_71, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_70, Release, 1)
      aie.use_lock(%lock_3_5_71, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_70, Release, 1)
      aie.use_lock(%lock_3_5_71, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_70, Release, 1)
      aie.use_lock(%lock_3_5_71, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf288, %buf287) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf285 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_70, Release, 1)
      aie.use_lock(%lock_3_5_71, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_69, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf287, %buf288, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf290, %buf284, %buf283) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf283, %buf289) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf286, %buf289) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf291, %buf283, %buf284) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf284, %buf291) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf281 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_70, Release, 1)
      aie.use_lock(%lock_3_5_71, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_69, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf287, %buf288, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf290, %buf280, %buf279) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf279, %buf289) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf282, %buf289) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf291, %buf279, %buf280) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf280, %buf291) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf289 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf290 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf291 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_5_70, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_67, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf275 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_68, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf273 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_66, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf269 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_66, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf276) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf278) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf277) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_68, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_67, Release, 1)
      aie.use_lock(%lock_2_5_68, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_67, Release, 1)
      aie.use_lock(%lock_2_5_68, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf275, %buf274) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_67, Release, 1)
      aie.use_lock(%lock_2_5_68, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf272 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_67, Release, 1)
      aie.use_lock(%lock_2_5_68, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_66, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf274, %buf275, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf277, %buf271, %buf270) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf270, %buf276) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf273, %buf276) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf278, %buf270, %buf271) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf271, %buf278) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf268 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_67, Release, 1)
      aie.use_lock(%lock_2_5_68, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_66, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf274, %buf275, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf277, %buf267, %buf266) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf266, %buf276) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf269, %buf276) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf278, %buf266, %buf267) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf267, %buf278) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf276 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf277 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf278 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_5_67, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_64, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf262 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_65, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf260 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_63, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf256 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_63, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf263) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf265) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf264) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_64, Release, 1)
      aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf262, %buf261) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_64, Release, 1)
      aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_64, Release, 1)
      aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf259 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_64, Release, 1)
      aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_63, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf261, %buf262, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf264, %buf258, %buf257) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf257, %buf263) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf260, %buf263) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf265, %buf257, %buf258) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf258, %buf265) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf255 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_64, Release, 1)
      aie.use_lock(%lock_1_5_65, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_63, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf261, %buf262, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf264, %buf254, %buf253) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf253, %buf263) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf256, %buf263) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf265, %buf253, %buf254) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf254, %buf265) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf263 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf264 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf265 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_5_64, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_61, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf249 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_62, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf247 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_60, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf243 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_60, Release, 1)
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
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf250) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf252) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf251) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_62, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf249, %buf248) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_61, Release, 1)
      aie.use_lock(%lock_0_5_62, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_61, Release, 1)
      aie.use_lock(%lock_0_5_62, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_61, Release, 1)
      aie.use_lock(%lock_0_5_62, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf246 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_61, Release, 1)
      aie.use_lock(%lock_0_5_62, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_60, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf248, %buf249, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf251, %buf245, %buf244) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf244, %buf250) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf247, %buf250) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf252, %buf244, %buf245) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf245, %buf252) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf242 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_61, Release, 1)
      aie.use_lock(%lock_0_5_62, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_60, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf248, %buf249, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf251, %buf241, %buf240) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf240, %buf250) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf243, %buf250) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf252, %buf240, %buf241) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf241, %buf252) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf250 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_74 = memref.collapse_shape %buf251 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_75 = memref.collapse_shape %buf252 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_5_61, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_58, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf236 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_59, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_57, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_57, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_4 = aie.core(%tile_3_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf237) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf239) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf238) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_59, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_58, Release, 1)
      aie.use_lock(%lock_3_4_59, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_58, Release, 1)
      aie.use_lock(%lock_3_4_59, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_58, Release, 1)
      aie.use_lock(%lock_3_4_59, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf236, %buf235) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf233 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_58, Release, 1)
      aie.use_lock(%lock_3_4_59, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_57, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf235, %buf236, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf238, %buf232, %buf231) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf231, %buf237) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf234, %buf237) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf239, %buf231, %buf232) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf232, %buf239) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf229 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_58, Release, 1)
      aie.use_lock(%lock_3_4_59, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_57, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf235, %buf236, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf238, %buf228, %buf227) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf227, %buf237) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf230, %buf237) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf239, %buf227, %buf228) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf228, %buf239) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf225 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf224 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_76 = memref.collapse_shape %buf238 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_76[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_4_58, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_55, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf216 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_56, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf214 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_54, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf210 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_54, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_4 = aie.core(%tile_2_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf217) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf219) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf218) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_56, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_55, Release, 1)
      aie.use_lock(%lock_2_4_56, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_55, Release, 1)
      aie.use_lock(%lock_2_4_56, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf216, %buf215) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_55, Release, 1)
      aie.use_lock(%lock_2_4_56, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf213 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_55, Release, 1)
      aie.use_lock(%lock_2_4_56, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_54, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf215, %buf216, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf218, %buf212, %buf211) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf211, %buf217) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf214, %buf217) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf219, %buf211, %buf212) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf212, %buf219) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf209 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_55, Release, 1)
      aie.use_lock(%lock_2_4_56, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_54, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf215, %buf216, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf218, %buf208, %buf207) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf207, %buf217) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf210, %buf217) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf219, %buf207, %buf208) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf208, %buf219) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf206 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf205 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf204 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_76 = memref.collapse_shape %buf218 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_76[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_4_55, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_52, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf196 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_53, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf194 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_51, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf190 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_51, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_4 = aie.core(%tile_1_4) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf197) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf199) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf198) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_53, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_52, Release, 1)
      aie.use_lock(%lock_1_4_53, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf196, %buf195) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_52, Release, 1)
      aie.use_lock(%lock_1_4_53, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_52, Release, 1)
      aie.use_lock(%lock_1_4_53, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf193 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_52, Release, 1)
      aie.use_lock(%lock_1_4_53, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_51, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf195, %buf196, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf198, %buf192, %buf191) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf191, %buf197) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf194, %buf197) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf199, %buf191, %buf192) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf192, %buf199) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf189 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_52, Release, 1)
      aie.use_lock(%lock_1_4_53, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_51, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf195, %buf196, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf198, %buf188, %buf187) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf187, %buf197) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf190, %buf197) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf199, %buf187, %buf188) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf188, %buf199) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf186 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf184 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_76 = memref.collapse_shape %buf198 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_76[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_4_52, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_49, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf176 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_50, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf174 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_48, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf170 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_48, Release, 1)
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
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf177) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf179) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf178) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_50, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf176, %buf175) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_49, Release, 1)
      aie.use_lock(%lock_0_4_50, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_49, Release, 1)
      aie.use_lock(%lock_0_4_50, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_49, Release, 1)
      aie.use_lock(%lock_0_4_50, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf173 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_49, Release, 1)
      aie.use_lock(%lock_0_4_50, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_48, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf175, %buf176, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf178, %buf172, %buf171) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf171, %buf177) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf174, %buf177) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf179, %buf171, %buf172) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf172, %buf179) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_49, Release, 1)
      aie.use_lock(%lock_0_4_50, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_48, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf175, %buf176, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf178, %buf168, %buf167) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf167, %buf177) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf170, %buf177) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf179, %buf167, %buf168) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf168, %buf179) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf166 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf165 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf164 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_76 = memref.collapse_shape %buf178 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_76[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_4_49, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_46, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf156 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_47, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf154 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_45, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf150 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_45, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf157) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf159) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf158) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_47, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_46, Release, 1)
      aie.use_lock(%lock_3_3_47, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_46, Release, 1)
      aie.use_lock(%lock_3_3_47, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_46, Release, 1)
      aie.use_lock(%lock_3_3_47, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf156, %buf155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_46, Release, 1)
      aie.use_lock(%lock_3_3_47, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_45, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf155, %buf156, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf158, %buf152, %buf151) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf151, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf154, %buf157) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf159, %buf151, %buf152) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf152, %buf159) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf149 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_46, Release, 1)
      aie.use_lock(%lock_3_3_47, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_45, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf155, %buf156, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf158, %buf148, %buf147) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf147, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf150, %buf157) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf159, %buf147, %buf148) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf148, %buf159) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf146 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf145 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf144 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_76 = memref.collapse_shape %buf158 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_76[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_3_3_46, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_43, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf136 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_44, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf134 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_42, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf130 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_42, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf137) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf139) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf138) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_43, Release, 1)
      aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_43, Release, 1)
      aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf136, %buf135) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_43, Release, 1)
      aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf133 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_43, Release, 1)
      aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_42, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf135, %buf136, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf138, %buf132, %buf131) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf131, %buf137) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf134, %buf137) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf139, %buf131, %buf132) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf132, %buf139) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf129 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_43, Release, 1)
      aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_42, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf135, %buf136, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf138, %buf128, %buf127) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf127, %buf137) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf130, %buf137) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf139, %buf127, %buf128) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf128, %buf139) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf126 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf125 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf124 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_76 = memref.collapse_shape %buf138 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_76[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_2_3_43, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_40, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf116 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_41, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf114 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_39, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf110 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_39, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf117) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf119) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf118) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_40, Release, 1)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf116, %buf115) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_40, Release, 1)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_40, Release, 1)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf113 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_40, Release, 1)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf115, %buf116, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf118, %buf112, %buf111) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf111, %buf117) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf114, %buf117) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf119, %buf111, %buf112) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf112, %buf119) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf109 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_40, Release, 1)
      aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf115, %buf116, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf118, %buf108, %buf107) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf107, %buf117) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf110, %buf117) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf119, %buf107, %buf108) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf108, %buf119) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf106 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf105 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf104 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_76 = memref.collapse_shape %buf118 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_76[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_1_3_40, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_37, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf96 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf94 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_36, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_36, Release, 1)
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
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf97) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf99) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf98) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf96, %buf95) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_37, Release, 1)
      aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_37, Release, 1)
      aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_37, Release, 1)
      aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf93 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_37, Release, 1)
      aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_36, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf95, %buf96, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf98, %buf92, %buf91) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf91, %buf97) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf94, %buf97) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf99, %buf91, %buf92) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf92, %buf99) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_37, Release, 1)
      aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_36, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf95, %buf96, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf98, %buf88, %buf87) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf87, %buf97) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf90, %buf97) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf99, %buf87, %buf88) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf88, %buf99) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf86 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf85 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf84 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_76 = memref.collapse_shape %buf98 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_76[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      aie.use_lock(%lock_0_3_37, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf66 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_34, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_32, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_33, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_31, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf70 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_31, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_34, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf77) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf79) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf78) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_32, Release, 1)
      aie.use_lock(%lock_3_2_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_32, Release, 1)
      aie.use_lock(%lock_3_2_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_32, Release, 1)
      aie.use_lock(%lock_3_2_33, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf76, %buf75) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      %collapse_shape = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_32, Release, 1)
      aie.use_lock(%lock_3_2_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_31, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf75, %buf76, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf78, %buf72, %buf71) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf71, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf74, %buf77) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf79, %buf71, %buf72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf72, %buf79) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf69 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_32, Release, 1)
      aie.use_lock(%lock_3_2_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_31, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf75, %buf76, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf78, %buf68, %buf67) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf67, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf70, %buf77) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf79, %buf67, %buf68) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf68, %buf79) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf66 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf65 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf64 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_3_2_35, Release, 1)
      aie.use_lock(%lock_3_2_32, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf46 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_29, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_27, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf56 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_28, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf54 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_26, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf50 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_26, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_29, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf57) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf59) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf58) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_28, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_27, Release, 1)
      aie.use_lock(%lock_2_2_28, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_27, Release, 1)
      aie.use_lock(%lock_2_2_28, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf56, %buf55) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_27, Release, 1)
      aie.use_lock(%lock_2_2_28, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf53 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_27, Release, 1)
      aie.use_lock(%lock_2_2_28, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_26, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf55, %buf56, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf58, %buf52, %buf51) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf51, %buf57) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf54, %buf57) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf59, %buf51, %buf52) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf52, %buf59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf49 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_27, Release, 1)
      aie.use_lock(%lock_2_2_28, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_26, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf55, %buf56, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf58, %buf48, %buf47) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf47, %buf57) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf50, %buf57) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf59, %buf47, %buf48) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf48, %buf59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf46 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf45 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf44 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_2_2_30, Release, 1)
      aie.use_lock(%lock_2_2_27, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_25, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_24, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf36 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_23, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf34 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_21, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_21, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_24, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf37) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf39) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf38) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_23, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_22, Release, 1)
      aie.use_lock(%lock_1_2_23, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf36, %buf35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_22, Release, 1)
      aie.use_lock(%lock_1_2_23, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_22, Release, 1)
      aie.use_lock(%lock_1_2_23, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf33 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_22, Release, 1)
      aie.use_lock(%lock_1_2_23, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_21, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf35, %buf36, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf38, %buf32, %buf31) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf31, %buf37) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf34, %buf37) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf39, %buf31, %buf32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf32, %buf39) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf29 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_22, Release, 1)
      aie.use_lock(%lock_1_2_23, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_21, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf35, %buf36, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf38, %buf28, %buf27) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf27, %buf37) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf30, %buf37) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf39, %buf27, %buf28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf28, %buf39) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf26 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf25 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf24 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_1_2_25, Release, 1)
      aie.use_lock(%lock_1_2_22, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_19, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf16 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_18, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_16, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_16, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_19, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf17) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf19) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf18) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf16, %buf15) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_17, Release, 1)
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_17, Release, 1)
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_17, Release, 1)
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf13 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_17, Release, 1)
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_16, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf15, %buf16, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape, %buf18, %buf12, %buf11) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf11, %buf17) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape, %buf14, %buf17) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf19, %buf11, %buf12) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf12, %buf19) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2, Release, 1)
      %collapse_shape_72 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      func.call @zero_fill_g_bf16(%collapse_shape_72) : (memref<4096xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_17, Release, 1)
      aie.use_lock(%lock_0_2_18, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_16, AcquireGreaterEqual, 1)
      func.call @matmul_a_b_bf16(%buf15, %buf16, %collapse_shape_72) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
      func.call @fused_softmax(%collapse_shape_72, %buf18, %buf8, %buf7) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @mul_r_gp(%buf7, %buf17) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @matmul_g_b_bf16(%collapse_shape_72, %buf10, %buf17) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      func.call @accum_sp_r_s(%buf19, %buf7, %buf8) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      func.call @vector_copy_32elems(%c0_i32, %buf8, %buf19) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2, Release, 1)
      %collapse_shape_73 = memref.collapse_shape %buf6 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape_73[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_74 = memref.collapse_shape %buf5 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_74[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_75 = memref.collapse_shape %buf4 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_75[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_0_2_20, Release, 1)
      aie.use_lock(%lock_0_2_17, Release, 1)
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
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %mem_tile_7_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_3_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 0)
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
      aie.use_lock(%lock_0_1_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf303 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf303 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_15, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf302 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf302 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_14, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_13, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf301 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf301 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_13, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf300 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf300 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_12, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_4_1_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf299 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_10, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_4_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb5, ^bb3)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_4_1_10, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf299 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_11, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf298 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_9, Release, 1)
      aie.next_bd ^bb5
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_5_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf297 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_7, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_5_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf296 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb5, ^bb3)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_5_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf297 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_8, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf296 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_6, Release, 1)
      aie.next_bd ^bb5
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_6_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf295 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_4, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_6_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf294 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb5, ^bb3)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_6_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf295 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_5, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf294 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_3, Release, 1)
      aie.next_bd ^bb5
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_7_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf293 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_1, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf292 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb5, ^bb3)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_7_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf293 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_2, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf292 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb5
    }
    aie.shim_dma_allocation @air_channel_0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_QK2L1_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_QK2L1_1(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @air_QK2L1_2(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_QK2L1_3(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @air_VIn_0(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_1(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_2(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_3(%shim_noc_tile_7_0, MM2S, 0)
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 4 : i64, id = 17 : i64, location = 4 : i64, row = -1 : i64}, {channel = 2 : i64, col = 4 : i64, id = 19 : i64, location = 4 : i64, row = -1 : i64}, {channel = 2 : i64, col = 5 : i64, id = 21 : i64, location = 5 : i64, row = -1 : i64}, {channel = 2 : i64, col = 5 : i64, id = 23 : i64, location = 5 : i64, row = -1 : i64}, {channel = 2 : i64, col = 6 : i64, id = 25 : i64, location = 6 : i64, row = -1 : i64}, {channel = 2 : i64, col = 6 : i64, id = 27 : i64, location = 6 : i64, row = -1 : i64}, {channel = 2 : i64, col = 7 : i64, id = 29 : i64, location = 7 : i64, row = -1 : i64}, {channel = 2 : i64, col = 7 : i64, id = 31 : i64, location = 7 : i64, row = -1 : i64}], sym_name = "attn_seg"}{
      airrt.herd_metadata {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 41 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 45 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 49 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 53 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 57 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 65 : i64, location = 0 : i64, row = 0 : i64}, {channel = 3 : i64, col = 0 : i64, id = 42 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 46 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 50 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 54 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 58 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 66 : i64, location = 0 : i64, row = 1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 43 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 47 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 51 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 55 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 59 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 67 : i64, location = 1 : i64, row = 2 : i64}, {channel = 3 : i64, col = 0 : i64, id = 44 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 48 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 52 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 56 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 60 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 68 : i64, location = 1 : i64, row = 3 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
  }
  air.channel @channel_0 [4, 1]
  air.channel @QK2L1_0 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_1 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_2 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_3 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @V2L1_0 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_0 [1]
  air.channel @V2L1_1 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_1 [1]
  air.channel @V2L1_2 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_2 [1]
  air.channel @V2L1_3 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_3 [1]
  air.channel @cascade_gp [4, 3] {channel_type = "cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  func.func @attention_bf16(%arg0: memref<256x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<256x64xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = airrt.wait_all : !airrt.event
    affine.for %arg4 = 0 to 1 {
      %p = airrt.segment_load "attn_seg" : i64
      %c12288 = arith.constant 12288 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c8192 = arith.constant 8192 : index
      %c16384 = arith.constant 16384 : index
      %c24576 = arith.constant 24576 : index
      %c3 = arith.constant 3 : index
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %c0_i64 = arith.constant 0 : i64
      %c0_2 = arith.constant 0 : index
      %c1_3 = arith.constant 1 : index
      %c41_i32 = arith.constant 41 : i32
      %1 = arith.index_cast %arg4 : index to i64
      %2 = arith.index_cast %c0_0 : index to i64
      %3 = arith.index_cast %c0_0 : index to i64
      %4 = arith.index_cast %c0_0 : index to i64
      %5 = arith.index_cast %c0_0 : index to i64
      %6 = arith.index_cast %c4096 : index to i64
      %7 = arith.index_cast %c8 : index to i64
      %8 = arith.index_cast %c64 : index to i64
      %9 = arith.index_cast %c1_1 : index to i64
      %10 = arith.index_cast %c4 : index to i64
      %11 = arith.index_cast %c8 : index to i64
      %12 = arith.index_cast %c64 : index to i64
      %13 = arith.index_cast %c8 : index to i64
      %14 = airrt.dma_memcpy_nd(%c41_i32, %1, %c0_i64, %arg0[%2, %3, %4, %5], [%10, %11, %12, %13], [%6, %7, %8, %9]) {chan_name = @QK2L1_0, metadata = @air_QK2L1_0} : (i32, i64, i64, memref<256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %15 = airrt.wait_all %14 : !airrt.event
      %c0_i64_4 = arith.constant 0 : i64
      %c0_5 = arith.constant 0 : index
      %c1_6 = arith.constant 1 : index
      %c41_i32_7 = arith.constant 41 : i32
      %16 = arith.index_cast %arg4 : index to i64
      %17 = arith.index_cast %c0_0 : index to i64
      %18 = arith.index_cast %c0_0 : index to i64
      %19 = arith.index_cast %c0_0 : index to i64
      %20 = arith.index_cast %c0_0 : index to i64
      %21 = arith.index_cast %c4096 : index to i64
      %22 = arith.index_cast %c8 : index to i64
      %23 = arith.index_cast %c64 : index to i64
      %24 = arith.index_cast %c1_1 : index to i64
      %25 = arith.index_cast %c2 : index to i64
      %26 = arith.index_cast %c8 : index to i64
      %27 = arith.index_cast %c64 : index to i64
      %28 = arith.index_cast %c8 : index to i64
      %29 = airrt.dma_memcpy_nd(%c41_i32_7, %16, %c0_i64_4, %arg1[%17, %18, %19, %20], [%25, %26, %27, %28], [%21, %22, %23, %24]) {chan_name = @QK2L1_0, metadata = @air_QK2L1_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %30 = airrt.wait_all %29 : !airrt.event
      %c0_i64_8 = arith.constant 0 : i64
      %c0_9 = arith.constant 0 : index
      %c1_10 = arith.constant 1 : index
      %c42_i32 = arith.constant 42 : i32
      %31 = arith.index_cast %arg4 : index to i64
      %32 = arith.index_cast %c0_0 : index to i64
      %33 = arith.index_cast %c0_0 : index to i64
      %34 = arith.index_cast %c0_0 : index to i64
      %35 = arith.index_cast %c0_0 : index to i64
      %36 = arith.index_cast %c4096 : index to i64
      %37 = arith.index_cast %c8 : index to i64
      %38 = arith.index_cast %c64 : index to i64
      %39 = arith.index_cast %c1_1 : index to i64
      %40 = arith.index_cast %c4 : index to i64
      %41 = arith.index_cast %c8 : index to i64
      %42 = arith.index_cast %c64 : index to i64
      %43 = arith.index_cast %c8 : index to i64
      %44 = airrt.dma_memcpy_nd(%c42_i32, %31, %c0_i64_8, %arg0[%32, %33, %34, %35], [%40, %41, %42, %43], [%36, %37, %38, %39]) {chan_name = @QK2L1_1, metadata = @air_QK2L1_1} : (i32, i64, i64, memref<256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %45 = airrt.wait_all %44 : !airrt.event
      %c0_i64_11 = arith.constant 0 : i64
      %c0_12 = arith.constant 0 : index
      %c1_13 = arith.constant 1 : index
      %c42_i32_14 = arith.constant 42 : i32
      %46 = arith.index_cast %arg4 : index to i64
      %47 = arith.index_cast %c0_0 : index to i64
      %48 = arith.index_cast %c0_0 : index to i64
      %49 = arith.index_cast %c0_0 : index to i64
      %50 = arith.index_cast %c8192 : index to i64
      %51 = arith.index_cast %c4096 : index to i64
      %52 = arith.index_cast %c8 : index to i64
      %53 = arith.index_cast %c64 : index to i64
      %54 = arith.index_cast %c1_1 : index to i64
      %55 = arith.index_cast %c2 : index to i64
      %56 = arith.index_cast %c8 : index to i64
      %57 = arith.index_cast %c64 : index to i64
      %58 = arith.index_cast %c8 : index to i64
      %59 = airrt.dma_memcpy_nd(%c42_i32_14, %46, %c0_i64_11, %arg1[%47, %48, %49, %50], [%55, %56, %57, %58], [%51, %52, %53, %54]) {chan_name = @QK2L1_1, metadata = @air_QK2L1_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %60 = airrt.wait_all %59 : !airrt.event
      %c0_i64_15 = arith.constant 0 : i64
      %c0_16 = arith.constant 0 : index
      %c1_17 = arith.constant 1 : index
      %c43_i32 = arith.constant 43 : i32
      %61 = arith.index_cast %arg4 : index to i64
      %62 = arith.index_cast %c0_0 : index to i64
      %63 = arith.index_cast %c0_0 : index to i64
      %64 = arith.index_cast %c0_0 : index to i64
      %65 = arith.index_cast %c0_0 : index to i64
      %66 = arith.index_cast %c4096 : index to i64
      %67 = arith.index_cast %c8 : index to i64
      %68 = arith.index_cast %c64 : index to i64
      %69 = arith.index_cast %c1_1 : index to i64
      %70 = arith.index_cast %c4 : index to i64
      %71 = arith.index_cast %c8 : index to i64
      %72 = arith.index_cast %c64 : index to i64
      %73 = arith.index_cast %c8 : index to i64
      %74 = airrt.dma_memcpy_nd(%c43_i32, %61, %c0_i64_15, %arg0[%62, %63, %64, %65], [%70, %71, %72, %73], [%66, %67, %68, %69]) {chan_name = @QK2L1_2, metadata = @air_QK2L1_2} : (i32, i64, i64, memref<256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %75 = airrt.wait_all %74 : !airrt.event
      %c0_i64_18 = arith.constant 0 : i64
      %c0_19 = arith.constant 0 : index
      %c1_20 = arith.constant 1 : index
      %c43_i32_21 = arith.constant 43 : i32
      %76 = arith.index_cast %arg4 : index to i64
      %77 = arith.index_cast %c0_0 : index to i64
      %78 = arith.index_cast %c0_0 : index to i64
      %79 = arith.index_cast %c0_0 : index to i64
      %80 = arith.index_cast %c16384 : index to i64
      %81 = arith.index_cast %c4096 : index to i64
      %82 = arith.index_cast %c8 : index to i64
      %83 = arith.index_cast %c64 : index to i64
      %84 = arith.index_cast %c1_1 : index to i64
      %85 = arith.index_cast %c2 : index to i64
      %86 = arith.index_cast %c8 : index to i64
      %87 = arith.index_cast %c64 : index to i64
      %88 = arith.index_cast %c8 : index to i64
      %89 = airrt.dma_memcpy_nd(%c43_i32_21, %76, %c0_i64_18, %arg1[%77, %78, %79, %80], [%85, %86, %87, %88], [%81, %82, %83, %84]) {chan_name = @QK2L1_2, metadata = @air_QK2L1_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %90 = airrt.wait_all %89 : !airrt.event
      %c0_i64_22 = arith.constant 0 : i64
      %c0_23 = arith.constant 0 : index
      %c1_24 = arith.constant 1 : index
      %c44_i32 = arith.constant 44 : i32
      %91 = arith.index_cast %arg4 : index to i64
      %92 = arith.index_cast %c0_0 : index to i64
      %93 = arith.index_cast %c0_0 : index to i64
      %94 = arith.index_cast %c0_0 : index to i64
      %95 = arith.index_cast %c0_0 : index to i64
      %96 = arith.index_cast %c4096 : index to i64
      %97 = arith.index_cast %c8 : index to i64
      %98 = arith.index_cast %c64 : index to i64
      %99 = arith.index_cast %c1_1 : index to i64
      %100 = arith.index_cast %c4 : index to i64
      %101 = arith.index_cast %c8 : index to i64
      %102 = arith.index_cast %c64 : index to i64
      %103 = arith.index_cast %c8 : index to i64
      %104 = airrt.dma_memcpy_nd(%c44_i32, %91, %c0_i64_22, %arg0[%92, %93, %94, %95], [%100, %101, %102, %103], [%96, %97, %98, %99]) {chan_name = @QK2L1_3, metadata = @air_QK2L1_3} : (i32, i64, i64, memref<256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %105 = airrt.wait_all %104 : !airrt.event
      %c0_i64_25 = arith.constant 0 : i64
      %c0_26 = arith.constant 0 : index
      %c1_27 = arith.constant 1 : index
      %c44_i32_28 = arith.constant 44 : i32
      %106 = arith.index_cast %arg4 : index to i64
      %107 = arith.index_cast %c0_0 : index to i64
      %108 = arith.index_cast %c0_0 : index to i64
      %109 = arith.index_cast %c0_0 : index to i64
      %110 = arith.index_cast %c24576 : index to i64
      %111 = arith.index_cast %c4096 : index to i64
      %112 = arith.index_cast %c8 : index to i64
      %113 = arith.index_cast %c64 : index to i64
      %114 = arith.index_cast %c1_1 : index to i64
      %115 = arith.index_cast %c2 : index to i64
      %116 = arith.index_cast %c8 : index to i64
      %117 = arith.index_cast %c64 : index to i64
      %118 = arith.index_cast %c8 : index to i64
      %119 = airrt.dma_memcpy_nd(%c44_i32_28, %106, %c0_i64_25, %arg1[%107, %108, %109, %110], [%115, %116, %117, %118], [%111, %112, %113, %114]) {chan_name = @QK2L1_3, metadata = @air_QK2L1_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %120 = airrt.wait_all %119 : !airrt.event
      %c0_i64_29 = arith.constant 0 : i64
      %c0_30 = arith.constant 0 : index
      %c1_31 = arith.constant 1 : index
      %c17_i32 = arith.constant 17 : i32
      %121 = arith.index_cast %arg4 : index to i64
      %122 = arith.index_cast %c0_30 : index to i64
      %123 = arith.index_cast %c0_30 : index to i64
      %124 = arith.index_cast %c0_30 : index to i64
      %125 = arith.index_cast %c0_0 : index to i64
      %126 = arith.index_cast %c0_30 : index to i64
      %127 = arith.index_cast %c0_30 : index to i64
      %128 = arith.index_cast %c0_30 : index to i64
      %129 = arith.index_cast %c1_1 : index to i64
      %130 = arith.index_cast %c1_31 : index to i64
      %131 = arith.index_cast %c1_31 : index to i64
      %132 = arith.index_cast %c1_31 : index to i64
      %133 = arith.index_cast %c8192 : index to i64
      %134 = airrt.dma_memcpy_nd(%c17_i32, %121, %c0_i64_29, %arg2[%122, %123, %124, %125], [%130, %131, %132, %133], [%126, %127, %128, %129]) {chan_name = @VIn_0, metadata = @air_VIn_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %135 = airrt.wait_all %134 : !airrt.event
      %c0_i64_32 = arith.constant 0 : i64
      %c0_33 = arith.constant 0 : index
      %c1_34 = arith.constant 1 : index
      %c21_i32 = arith.constant 21 : i32
      %136 = arith.index_cast %arg4 : index to i64
      %137 = arith.index_cast %c0_33 : index to i64
      %138 = arith.index_cast %c0_33 : index to i64
      %139 = arith.index_cast %c0_33 : index to i64
      %140 = arith.index_cast %c8192 : index to i64
      %141 = arith.index_cast %c0_33 : index to i64
      %142 = arith.index_cast %c0_33 : index to i64
      %143 = arith.index_cast %c0_33 : index to i64
      %144 = arith.index_cast %c1_1 : index to i64
      %145 = arith.index_cast %c1_34 : index to i64
      %146 = arith.index_cast %c1_34 : index to i64
      %147 = arith.index_cast %c1_34 : index to i64
      %148 = arith.index_cast %c8192 : index to i64
      %149 = airrt.dma_memcpy_nd(%c21_i32, %136, %c0_i64_32, %arg2[%137, %138, %139, %140], [%145, %146, %147, %148], [%141, %142, %143, %144]) {chan_name = @VIn_1, metadata = @air_VIn_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %150 = airrt.wait_all %149 : !airrt.event
      %c0_i64_35 = arith.constant 0 : i64
      %c0_36 = arith.constant 0 : index
      %c1_37 = arith.constant 1 : index
      %c25_i32 = arith.constant 25 : i32
      %151 = arith.index_cast %arg4 : index to i64
      %152 = arith.index_cast %c0_36 : index to i64
      %153 = arith.index_cast %c0_36 : index to i64
      %154 = arith.index_cast %c0_36 : index to i64
      %155 = arith.index_cast %c16384 : index to i64
      %156 = arith.index_cast %c0_36 : index to i64
      %157 = arith.index_cast %c0_36 : index to i64
      %158 = arith.index_cast %c0_36 : index to i64
      %159 = arith.index_cast %c1_1 : index to i64
      %160 = arith.index_cast %c1_37 : index to i64
      %161 = arith.index_cast %c1_37 : index to i64
      %162 = arith.index_cast %c1_37 : index to i64
      %163 = arith.index_cast %c8192 : index to i64
      %164 = airrt.dma_memcpy_nd(%c25_i32, %151, %c0_i64_35, %arg2[%152, %153, %154, %155], [%160, %161, %162, %163], [%156, %157, %158, %159]) {chan_name = @VIn_2, metadata = @air_VIn_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %165 = airrt.wait_all %164 : !airrt.event
      %c0_i64_38 = arith.constant 0 : i64
      %c0_39 = arith.constant 0 : index
      %c1_40 = arith.constant 1 : index
      %c29_i32 = arith.constant 29 : i32
      %166 = arith.index_cast %arg4 : index to i64
      %167 = arith.index_cast %c0_39 : index to i64
      %168 = arith.index_cast %c0_39 : index to i64
      %169 = arith.index_cast %c0_39 : index to i64
      %170 = arith.index_cast %c24576 : index to i64
      %171 = arith.index_cast %c0_39 : index to i64
      %172 = arith.index_cast %c0_39 : index to i64
      %173 = arith.index_cast %c0_39 : index to i64
      %174 = arith.index_cast %c1_1 : index to i64
      %175 = arith.index_cast %c1_40 : index to i64
      %176 = arith.index_cast %c1_40 : index to i64
      %177 = arith.index_cast %c1_40 : index to i64
      %178 = arith.index_cast %c8192 : index to i64
      %179 = airrt.dma_memcpy_nd(%c29_i32, %166, %c0_i64_38, %arg2[%167, %168, %169, %170], [%175, %176, %177, %178], [%171, %172, %173, %174]) {chan_name = @VIn_3, metadata = @air_VIn_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %180 = airrt.wait_all %179 : !airrt.event
      %c0_i64_41 = arith.constant 0 : i64
      %c0_42 = arith.constant 0 : index
      %c1_43 = arith.constant 1 : index
      %c37_i32 = arith.constant 37 : i32
      %181 = arith.index_cast %arg4 : index to i64
      %182 = arith.index_cast %c0_42 : index to i64
      %183 = arith.index_cast %c0_42 : index to i64
      %184 = arith.index_cast %c0_42 : index to i64
      %185 = arith.index_cast %c0_0 : index to i64
      %186 = arith.index_cast %c0_42 : index to i64
      %187 = arith.index_cast %c0_42 : index to i64
      %188 = arith.index_cast %c0_42 : index to i64
      %189 = arith.index_cast %c1_1 : index to i64
      %190 = arith.index_cast %c1_43 : index to i64
      %191 = arith.index_cast %c1_43 : index to i64
      %192 = arith.index_cast %c1_43 : index to i64
      %193 = arith.index_cast %c4096 : index to i64
      %194 = airrt.dma_memcpy_nd(%c37_i32, %181, %c0_i64_41, %arg3[%182, %183, %184, %185], [%190, %191, %192, %193], [%186, %187, %188, %189]) {chan_name = @channel_0, metadata = @air_channel_0_0} : (i32, i64, i64, memref<256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %195 = airrt.wait_all %194 : !airrt.event
      %c0_i64_44 = arith.constant 0 : i64
      %c0_45 = arith.constant 0 : index
      %c1_46 = arith.constant 1 : index
      %c37_i32_47 = arith.constant 37 : i32
      %196 = arith.index_cast %arg4 : index to i64
      %197 = arith.index_cast %c0_45 : index to i64
      %198 = arith.index_cast %c0_45 : index to i64
      %199 = arith.index_cast %c0_45 : index to i64
      %200 = arith.index_cast %c4096 : index to i64
      %201 = arith.index_cast %c0_45 : index to i64
      %202 = arith.index_cast %c0_45 : index to i64
      %203 = arith.index_cast %c0_45 : index to i64
      %204 = arith.index_cast %c1_1 : index to i64
      %205 = arith.index_cast %c1_46 : index to i64
      %206 = arith.index_cast %c1_46 : index to i64
      %207 = arith.index_cast %c1_46 : index to i64
      %208 = arith.index_cast %c4096 : index to i64
      %209 = airrt.dma_memcpy_nd(%c37_i32_47, %196, %c0_i64_44, %arg3[%197, %198, %199, %200], [%205, %206, %207, %208], [%201, %202, %203, %204]) {chan_name = @channel_0, metadata = @air_channel_0_1} : (i32, i64, i64, memref<256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %210 = airrt.wait_all %209 : !airrt.event
      %c0_i64_48 = arith.constant 0 : i64
      %c0_49 = arith.constant 0 : index
      %c1_50 = arith.constant 1 : index
      %c37_i32_51 = arith.constant 37 : i32
      %211 = arith.index_cast %arg4 : index to i64
      %212 = arith.index_cast %c0_49 : index to i64
      %213 = arith.index_cast %c0_49 : index to i64
      %214 = arith.index_cast %c0_49 : index to i64
      %215 = arith.index_cast %c8192 : index to i64
      %216 = arith.index_cast %c0_49 : index to i64
      %217 = arith.index_cast %c0_49 : index to i64
      %218 = arith.index_cast %c0_49 : index to i64
      %219 = arith.index_cast %c1_1 : index to i64
      %220 = arith.index_cast %c1_50 : index to i64
      %221 = arith.index_cast %c1_50 : index to i64
      %222 = arith.index_cast %c1_50 : index to i64
      %223 = arith.index_cast %c4096 : index to i64
      %224 = airrt.dma_memcpy_nd(%c37_i32_51, %211, %c0_i64_48, %arg3[%212, %213, %214, %215], [%220, %221, %222, %223], [%216, %217, %218, %219]) {chan_name = @channel_0, metadata = @air_channel_0_2} : (i32, i64, i64, memref<256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %225 = airrt.wait_all %224 : !airrt.event
      %c0_i64_52 = arith.constant 0 : i64
      %c0_53 = arith.constant 0 : index
      %c1_54 = arith.constant 1 : index
      %c37_i32_55 = arith.constant 37 : i32
      %226 = arith.index_cast %arg4 : index to i64
      %227 = arith.index_cast %c0_53 : index to i64
      %228 = arith.index_cast %c0_53 : index to i64
      %229 = arith.index_cast %c0_53 : index to i64
      %230 = arith.index_cast %c12288 : index to i64
      %231 = arith.index_cast %c0_53 : index to i64
      %232 = arith.index_cast %c0_53 : index to i64
      %233 = arith.index_cast %c0_53 : index to i64
      %234 = arith.index_cast %c1_1 : index to i64
      %235 = arith.index_cast %c1_54 : index to i64
      %236 = arith.index_cast %c1_54 : index to i64
      %237 = arith.index_cast %c1_54 : index to i64
      %238 = arith.index_cast %c4096 : index to i64
      %239 = airrt.dma_memcpy_nd(%c37_i32_55, %226, %c0_i64_52, %arg3[%227, %228, %229, %230], [%235, %236, %237, %238], [%231, %232, %233, %234]) {chan_name = @channel_0, metadata = @air_channel_0_3} : (i32, i64, i64, memref<256x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %240 = airrt.wait_all %239 : !airrt.event
      %c0_56 = arith.constant 0 : index
      %c1_57 = arith.constant 1 : index
      %241 = airrt.wait_all : !airrt.event
      affine.for %arg5 = 0 to 1 {
        affine.for %arg6 = 0 to 1 {
          %c3_58 = arith.constant 3 : index
          %c64_59 = arith.constant 64 : index
          %c8_60 = arith.constant 8 : index
          %c1_61 = arith.constant 1 : index
          %c2_62 = arith.constant 2 : index
          %c0_63 = arith.constant 0 : index
          %c4_64 = arith.constant 4 : index
          %242 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %243 = airrt.wait_all : !airrt.event
          %244 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %245 = airrt.wait_all : !airrt.event
          %246 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %247 = airrt.wait_all : !airrt.event
          %248 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %249 = airrt.wait_all : !airrt.event
          %250 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %251 = airrt.wait_all : !airrt.event
          %252 = airrt.wait_all %251 : !airrt.event
          %253 = airrt.wait_all %252 : !airrt.event
          airrt.dealloc %250 : memref<64x64xbf16, 1 : i32>
          %254 = airrt.wait_all : !airrt.event
          %255 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %256 = airrt.wait_all : !airrt.event
          %257 = airrt.wait_all %256 : !airrt.event
          %258 = airrt.wait_all %257 : !airrt.event
          airrt.dealloc %255 : memref<64x64xbf16, 1 : i32>
          %259 = airrt.wait_all : !airrt.event
          %260 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %261 = airrt.wait_all : !airrt.event
          %262 = airrt.wait_all %261 : !airrt.event
          %263 = airrt.wait_all %262 : !airrt.event
          airrt.dealloc %260 : memref<64x64xbf16, 1 : i32>
          %264 = airrt.wait_all : !airrt.event
          %265 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %266 = airrt.wait_all : !airrt.event
          %267 = airrt.wait_all %266 : !airrt.event
          %268 = airrt.wait_all %267 : !airrt.event
          airrt.dealloc %265 : memref<64x64xbf16, 1 : i32>
          %269 = airrt.wait_all : !airrt.event
          %270 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %271 = airrt.wait_all : !airrt.event
          %272 = airrt.wait_all %271 : !airrt.event
          %273 = airrt.wait_all %272 : !airrt.event
          airrt.dealloc %270 : memref<64x64xbf16, 1 : i32>
          %274 = airrt.wait_all : !airrt.event
          %275 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %276 = airrt.wait_all : !airrt.event
          %277 = airrt.wait_all %276 : !airrt.event
          %278 = airrt.wait_all %277 : !airrt.event
          airrt.dealloc %275 : memref<64x64xbf16, 1 : i32>
          %279 = airrt.wait_all : !airrt.event
          %280 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %281 = airrt.wait_all : !airrt.event
          %282 = airrt.wait_all %281 : !airrt.event
          %283 = airrt.wait_all %282 : !airrt.event
          airrt.dealloc %280 : memref<64x64xbf16, 1 : i32>
          %284 = airrt.wait_all : !airrt.event
          %285 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %286 = airrt.wait_all : !airrt.event
          %287 = airrt.wait_all %286 : !airrt.event
          %288 = airrt.wait_all %287 : !airrt.event
          airrt.dealloc %285 : memref<64x64xbf16, 1 : i32>
          %289 = airrt.wait_all : !airrt.event
          %290 = airrt.wait_all %243 : !airrt.event
          %291 = airrt.wait_all %245 : !airrt.event
          %292 = airrt.wait_all %247 : !airrt.event
          %293 = airrt.wait_all %249 : !airrt.event
          %294 = airrt.wait_all %290 : !airrt.event
          %295 = airrt.wait_all %291 : !airrt.event
          %296 = airrt.wait_all %292 : !airrt.event
          %297 = airrt.wait_all %293 : !airrt.event
          %h = airrt.herd_load "herd_0" () {segment_name = "attn_seg"} : () -> i64
          %298 = airrt.wait_all : !airrt.event
          airrt.dealloc %248 : memref<64x64xbf16, 1 : i32>
          %299 = airrt.wait_all : !airrt.event
          airrt.dealloc %246 : memref<64x64xbf16, 1 : i32>
          %300 = airrt.wait_all : !airrt.event
          airrt.dealloc %244 : memref<64x64xbf16, 1 : i32>
          %301 = airrt.wait_all : !airrt.event
          airrt.dealloc %242 : memref<64x64xbf16, 1 : i32>
          %302 = airrt.wait_all : !airrt.event
          airrt.wait_all %254, %259, %264, %269, %274, %279, %284, %289, %298, %299, %300, %301, %302 {air.segment_end}
        }
      }
      airrt.wait_all %150, %180, %210, %240, %30, %15, %45, %60, %90, %75, %105, %120, %241, %225, %195, %165, %135 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
