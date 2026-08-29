#loop_annotation = #llvm.loop_annotation<mustProgress = true>
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
    %lock_7_1 = aie.lock(%mem_tile_7_1, 1) {init = 1 : i32}
    %lock_7_1_0 = aie.lock(%mem_tile_7_1, 0) {init = 0 : i32}
    %lock_6_1 = aie.lock(%mem_tile_6_1, 1) {init = 1 : i32}
    %lock_6_1_1 = aie.lock(%mem_tile_6_1, 0) {init = 0 : i32}
    %lock_5_1 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %lock_5_1_2 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_4_1 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %lock_4_1_3 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_3_1 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_4 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_5 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_6 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_7 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_8 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_9 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_10 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_11 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_12 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_13 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_14 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_15 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_16 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_17 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_18 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_19 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_20 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_21 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_22 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_23 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_24 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_25 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_26 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_27 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_28 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_29 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_30 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_31 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_32 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_33 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_34 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_35 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_36 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_37 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_38 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_39 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 3) {init = 1 : i32}
    %lock_0_4_40 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_41 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_42 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 3) {init = 1 : i32}
    %lock_1_4_43 = aie.lock(%tile_1_4, 2) {init = 0 : i32}
    %lock_1_4_44 = aie.lock(%tile_1_4, 1) {init = 1 : i32}
    %lock_1_4_45 = aie.lock(%tile_1_4, 0) {init = 0 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 3) {init = 1 : i32}
    %lock_2_4_46 = aie.lock(%tile_2_4, 2) {init = 0 : i32}
    %lock_2_4_47 = aie.lock(%tile_2_4, 1) {init = 1 : i32}
    %lock_2_4_48 = aie.lock(%tile_2_4, 0) {init = 0 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 3) {init = 1 : i32}
    %lock_3_4_49 = aie.lock(%tile_3_4, 2) {init = 0 : i32}
    %lock_3_4_50 = aie.lock(%tile_3_4, 1) {init = 1 : i32}
    %lock_3_4_51 = aie.lock(%tile_3_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 3) {init = 1 : i32}
    %lock_0_5_52 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_53 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_54 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 3) {init = 1 : i32}
    %lock_1_5_55 = aie.lock(%tile_1_5, 2) {init = 0 : i32}
    %lock_1_5_56 = aie.lock(%tile_1_5, 1) {init = 1 : i32}
    %lock_1_5_57 = aie.lock(%tile_1_5, 0) {init = 0 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 3) {init = 1 : i32}
    %lock_2_5_58 = aie.lock(%tile_2_5, 2) {init = 0 : i32}
    %lock_2_5_59 = aie.lock(%tile_2_5, 1) {init = 1 : i32}
    %lock_2_5_60 = aie.lock(%tile_2_5, 0) {init = 0 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 3) {init = 1 : i32}
    %lock_3_5_61 = aie.lock(%tile_3_5, 2) {init = 0 : i32}
    %lock_3_5_62 = aie.lock(%tile_3_5, 1) {init = 1 : i32}
    %lock_3_5_63 = aie.lock(%tile_3_5, 0) {init = 0 : i32}
    %buf235 = aie.buffer(%mem_tile_0_1) {sym_name = "buf235"} : memref<64x64xbf16, 1 : i32> 
    %buf234 = aie.buffer(%mem_tile_1_1) {sym_name = "buf234"} : memref<64x64xbf16, 1 : i32> 
    %buf233 = aie.buffer(%mem_tile_2_1) {sym_name = "buf233"} : memref<64x64xbf16, 1 : i32> 
    %buf232 = aie.buffer(%mem_tile_3_1) {sym_name = "buf232"} : memref<64x64xbf16, 1 : i32> 
    %buf231 = aie.buffer(%mem_tile_4_1) {sym_name = "buf231"} : memref<64x64xbf16, 1 : i32> 
    %buf230 = aie.buffer(%mem_tile_5_1) {sym_name = "buf230"} : memref<64x64xbf16, 1 : i32> 
    %buf229 = aie.buffer(%mem_tile_6_1) {sym_name = "buf229"} : memref<64x64xbf16, 1 : i32> 
    %buf228 = aie.buffer(%mem_tile_7_1) {sym_name = "buf228"} : memref<64x64xbf16, 1 : i32> 
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
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<512x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<512x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<512x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<512x64xbf16>
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_5_62, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf224 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_63, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf222 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_5_61, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf225) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf227) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf226) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_62, Release, 1)
      aie.use_lock(%lock_3_5_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_62, Release, 1)
      aie.use_lock(%lock_3_5_63, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_5_62, Release, 1)
      aie.use_lock(%lock_3_5_63, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf224, %buf223) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_5_62, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_66 = memref.collapse_shape %buf221 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_66) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_63, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_5_61, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf223, %buf224, %collapse_shape_66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5_62, Release, 1)
        func.call @fused_softmax(%collapse_shape_66, %buf226, %buf220, %buf219) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf219, %buf225) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_66, %buf222, %buf225) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf227, %buf219, %buf220) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf220, %buf227) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf225 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_64 = memref.collapse_shape %buf226 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_65 = memref.collapse_shape %buf227 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_5_59, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf215 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf213 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_5_58, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf216) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf218) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf217) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_59, Release, 1)
      aie.use_lock(%lock_2_5_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_59, Release, 1)
      aie.use_lock(%lock_2_5_60, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf215, %buf214) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_5_59, Release, 1)
      aie.use_lock(%lock_2_5_60, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_5_59, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_66 = memref.collapse_shape %buf212 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_66) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_60, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_5_58, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf214, %buf215, %collapse_shape_66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5_59, Release, 1)
        func.call @fused_softmax(%collapse_shape_66, %buf217, %buf211, %buf210) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf210, %buf216) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_66, %buf213, %buf216) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf218, %buf210, %buf211) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf211, %buf218) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf216 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_64 = memref.collapse_shape %buf217 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_65 = memref.collapse_shape %buf218 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_5_56, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf206 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_57, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf204 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_5_55, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf207) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf209) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf208) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_56, Release, 1)
      aie.use_lock(%lock_1_5_57, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf206, %buf205) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_5_56, Release, 1)
      aie.use_lock(%lock_1_5_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_56, Release, 1)
      aie.use_lock(%lock_1_5_57, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_5_56, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_66 = memref.collapse_shape %buf203 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_66) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_57, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_5_55, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf205, %buf206, %collapse_shape_66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5_56, Release, 1)
        func.call @fused_softmax(%collapse_shape_66, %buf208, %buf202, %buf201) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf201, %buf207) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_66, %buf204, %buf207) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf209, %buf201, %buf202) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf202, %buf209) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf207 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_64 = memref.collapse_shape %buf208 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_65 = memref.collapse_shape %buf209 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_53, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf197 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_54, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf195 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_5_52, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf198) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf200) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf199) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_54, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf197, %buf196) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_5_53, Release, 1)
      aie.use_lock(%lock_0_5_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_53, Release, 1)
      aie.use_lock(%lock_0_5_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_53, Release, 1)
      aie.use_lock(%lock_0_5_54, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_53, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_66 = memref.collapse_shape %buf194 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_66) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_54, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_5_52, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf196, %buf197, %collapse_shape_66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5_53, Release, 1)
        func.call @fused_softmax(%collapse_shape_66, %buf199, %buf193, %buf192) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf192, %buf198) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_66, %buf195, %buf198) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf200, %buf192, %buf193) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf193, %buf200) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_5, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf198 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_64 = memref.collapse_shape %buf199 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_65 = memref.collapse_shape %buf200 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_4_50, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf188 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_51, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf186 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_4_49, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf189) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf191) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf190) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_50, Release, 1)
      aie.use_lock(%lock_3_4_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_50, Release, 1)
      aie.use_lock(%lock_3_4_51, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_4_50, Release, 1)
      aie.use_lock(%lock_3_4_51, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf188, %buf187) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_4_50, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_67 = memref.collapse_shape %buf185 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_67) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_4_49, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf187, %buf188, %collapse_shape_67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4_50, Release, 1)
        func.call @fused_softmax(%collapse_shape_67, %buf190, %buf184, %buf183) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf183, %buf189) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_67, %buf186, %buf189) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf191, %buf183, %buf184) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf184, %buf191) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf182 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf181 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf180 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_66 = memref.collapse_shape %buf190 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_66[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_4_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf172 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf170 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_4_46, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf173) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf175) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf174) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_47, Release, 1)
      aie.use_lock(%lock_2_4_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_47, Release, 1)
      aie.use_lock(%lock_2_4_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf172, %buf171) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_4_47, Release, 1)
      aie.use_lock(%lock_2_4_48, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_4_47, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_67 = memref.collapse_shape %buf169 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_67) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_48, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_4_46, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf171, %buf172, %collapse_shape_67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4_47, Release, 1)
        func.call @fused_softmax(%collapse_shape_67, %buf174, %buf168, %buf167) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf167, %buf173) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_67, %buf170, %buf173) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf175, %buf167, %buf168) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf168, %buf175) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf166 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf165 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf164 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_66 = memref.collapse_shape %buf174 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_66[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4_44, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf156 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_45, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf154 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_4_43, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf157) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf159) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf158) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_44, Release, 1)
      aie.use_lock(%lock_1_4_45, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf156, %buf155) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_4_44, Release, 1)
      aie.use_lock(%lock_1_4_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_44, Release, 1)
      aie.use_lock(%lock_1_4_45, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_4_44, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_67 = memref.collapse_shape %buf153 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_67) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_45, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_4_43, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf155, %buf156, %collapse_shape_67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4_44, Release, 1)
        func.call @fused_softmax(%collapse_shape_67, %buf158, %buf152, %buf151) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf151, %buf157) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_67, %buf154, %buf157) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf159, %buf151, %buf152) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf152, %buf159) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf150 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf149 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf148 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_66 = memref.collapse_shape %buf158 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_66[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_41, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf140 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_42, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf138 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_4_40, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf141) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf143) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf142) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_42, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf140, %buf139) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_4_41, Release, 1)
      aie.use_lock(%lock_0_4_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_41, Release, 1)
      aie.use_lock(%lock_0_4_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_41, Release, 1)
      aie.use_lock(%lock_0_4_42, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_41, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_67 = memref.collapse_shape %buf137 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_67) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_42, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_4_40, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf139, %buf140, %collapse_shape_67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4_41, Release, 1)
        func.call @fused_softmax(%collapse_shape_67, %buf142, %buf136, %buf135) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf135, %buf141) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_67, %buf138, %buf141) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf143, %buf135, %buf136) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf136, %buf143) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_4, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf134 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf133 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf132 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_66 = memref.collapse_shape %buf142 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_66[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_38, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf124 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_39, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf122 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_37, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf125) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf127) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf126) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_39, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_38, Release, 1)
      aie.use_lock(%lock_3_3_39, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_38, Release, 1)
      aie.use_lock(%lock_3_3_39, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_3_38, Release, 1)
      aie.use_lock(%lock_3_3_39, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf124, %buf123) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_38, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_67 = memref.collapse_shape %buf121 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_67) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_39, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3_37, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf123, %buf124, %collapse_shape_67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_38, Release, 1)
        func.call @fused_softmax(%collapse_shape_67, %buf126, %buf120, %buf119) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf119, %buf125) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_67, %buf122, %buf125) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf127, %buf119, %buf120) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf120, %buf127) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf118 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf117 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf116 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_66 = memref.collapse_shape %buf126 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_66[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_35, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf108 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_36, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf106 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_34, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf109) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf111) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf110) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_36, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_35, Release, 1)
      aie.use_lock(%lock_2_3_36, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_35, Release, 1)
      aie.use_lock(%lock_2_3_36, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf108, %buf107) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_35, Release, 1)
      aie.use_lock(%lock_2_3_36, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_3_35, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_67 = memref.collapse_shape %buf105 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_67) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3_34, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf107, %buf108, %collapse_shape_67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_35, Release, 1)
        func.call @fused_softmax(%collapse_shape_67, %buf110, %buf104, %buf103) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf103, %buf109) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_67, %buf106, %buf109) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf111, %buf103, %buf104) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf104, %buf111) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf102 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf101 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf100 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_66 = memref.collapse_shape %buf110 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_66[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_32, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf92 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_33, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf90 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_31, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf93) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf95) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf94) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_32, Release, 1)
      aie.use_lock(%lock_1_3_33, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf92, %buf91) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_32, Release, 1)
      aie.use_lock(%lock_1_3_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_32, Release, 1)
      aie.use_lock(%lock_1_3_33, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_3_32, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_67 = memref.collapse_shape %buf89 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_67) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_33, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3_31, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf91, %buf92, %collapse_shape_67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_32, Release, 1)
        func.call @fused_softmax(%collapse_shape_67, %buf94, %buf88, %buf87) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf87, %buf93) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_67, %buf90, %buf93) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf95, %buf87, %buf88) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf88, %buf95) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf86 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf85 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf84 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_66 = memref.collapse_shape %buf94 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_66[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_29, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_30, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_28, Release, 1)
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
      func.call @zero_fill_gp_bf16(%buf77) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf79) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf78) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_30, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf76, %buf75) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_29, Release, 1)
      aie.use_lock(%lock_0_3_30, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_29, Release, 1)
      aie.use_lock(%lock_0_3_30, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_29, Release, 1)
      aie.use_lock(%lock_0_3_30, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_29, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_67 = memref.collapse_shape %buf73 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_67) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_30, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_28, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf75, %buf76, %collapse_shape_67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_29, Release, 1)
        func.call @fused_softmax(%collapse_shape_67, %buf78, %buf72, %buf71) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf71, %buf77) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_67, %buf74, %buf77) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf79, %buf71, %buf72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf72, %buf79) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf70 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf69 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf68 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_66 = memref.collapse_shape %buf78 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_66[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf54 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_26, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf60 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_25, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf58 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_23, Release, 1)
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
      aie.use_lock(%lock_3_2_26, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf61) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf63) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf62) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_24, Release, 1)
      aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_24, Release, 1)
      aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_3_2_24, Release, 1)
      aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf60, %buf59) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_24, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_66 = memref.collapse_shape %buf57 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_66) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_25, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_23, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf59, %buf60, %collapse_shape_66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_24, Release, 1)
        func.call @fused_softmax(%collapse_shape_66, %buf62, %buf56, %buf55) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf55, %buf61) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_66, %buf58, %buf61) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf63, %buf55, %buf56) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf56, %buf63) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf54 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf53 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf52 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_3_2_27, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_22, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf38 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_21, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf44 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf42 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_18, Release, 1)
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
      aie.use_lock(%lock_2_2_21, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf45) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf47) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf46) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_19, Release, 1)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_19, Release, 1)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf44, %buf43) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_19, Release, 1)
      aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_2_2_19, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_66 = memref.collapse_shape %buf41 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_66) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_18, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf43, %buf44, %collapse_shape_66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_19, Release, 1)
        func.call @fused_softmax(%collapse_shape_66, %buf46, %buf40, %buf39) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf39, %buf45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_66, %buf42, %buf45) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf47, %buf39, %buf40) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf40, %buf47) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf38 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf37 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf36 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_2_2_22, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_17, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf22 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_16, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf28 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_15, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf26 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, 1)
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
      aie.use_lock(%lock_1_2_16, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf29) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf31) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf30) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_15, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_14, Release, 1)
      aie.use_lock(%lock_1_2_15, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf28, %buf27) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_14, Release, 1)
      aie.use_lock(%lock_1_2_15, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_14, Release, 1)
      aie.use_lock(%lock_1_2_15, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_1_2_14, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_66 = memref.collapse_shape %buf25 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_66) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_15, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf27, %buf28, %collapse_shape_66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_14, Release, 1)
        func.call @fused_softmax(%collapse_shape_66, %buf30, %buf24, %buf23) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf23, %buf29) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_66, %buf26, %buf29) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf31, %buf23, %buf24) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf24, %buf31) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf22 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf21 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf20 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_1_2_17, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_11, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_10, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_8, Release, 1)
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
      aie.use_lock(%lock_0_2_11, AcquireGreaterEqual, 1)
      func.call @zero_fill_gp_bf16(%buf13) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf15) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf14) : (memref<64x1xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_10, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf12, %buf11) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_9, Release, 1)
      aie.use_lock(%lock_0_2_10, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_9, Release, 1)
      aie.use_lock(%lock_0_2_10, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_9, Release, 1)
      aie.use_lock(%lock_0_2_10, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_9, Release, 1)
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_66 = memref.collapse_shape %buf9 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_66) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_10, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_8, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf11, %buf12, %collapse_shape_66) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_9, Release, 1)
        func.call @fused_softmax(%collapse_shape_66, %buf14, %buf8, %buf7) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf7, %buf13) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_66, %buf10, %buf13) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf15, %buf7, %buf8) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf8, %buf15) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf6 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_64 = memref.collapse_shape %buf5 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_64[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_65 = memref.collapse_shape %buf4 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_65[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
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
      aie.use_lock(%lock_0_2_12, Release, 1)
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
      aie.use_lock(%lock_0_1_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf235 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf235 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_7, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf234 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_6, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf233 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf233 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_5, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf232 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf232 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_4, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf231 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf231 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_3, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf230 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_2, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_6_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf229 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_6_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf229 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_6_1_1, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_7_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf228 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_7_1_0, Release, 1)
      aie.next_bd ^bb4
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
    airrt.segment_metadata attributes {dma_allocations = [{channel = 2 : i64, col = 4 : i64, id = 17 : i64, location = 4 : i64, row = -1 : i64}, {channel = 2 : i64, col = 5 : i64, id = 19 : i64, location = 5 : i64, row = -1 : i64}, {channel = 2 : i64, col = 6 : i64, id = 21 : i64, location = 6 : i64, row = -1 : i64}, {channel = 2 : i64, col = 7 : i64, id = 23 : i64, location = 7 : i64, row = -1 : i64}], sym_name = "attn_seg"}{
      airrt.herd_metadata {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 33 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 37 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 41 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 45 : i64, location = 0 : i64, row = 0 : i64}, {channel = 2 : i64, col = 0 : i64, id = 49 : i64, location = 0 : i64, row = 0 : i64}, {channel = 3 : i64, col = 0 : i64, id = 34 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 38 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 42 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 46 : i64, location = 0 : i64, row = 1 : i64}, {channel = 3 : i64, col = 0 : i64, id = 50 : i64, location = 0 : i64, row = 1 : i64}, {channel = 2 : i64, col = 0 : i64, id = 35 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 39 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 43 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 47 : i64, location = 1 : i64, row = 2 : i64}, {channel = 2 : i64, col = 0 : i64, id = 51 : i64, location = 1 : i64, row = 2 : i64}, {channel = 3 : i64, col = 0 : i64, id = 36 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 40 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 44 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 48 : i64, location = 1 : i64, row = 3 : i64}, {channel = 3 : i64, col = 0 : i64, id = 52 : i64, location = 1 : i64, row = 3 : i64}], loc_x = 0 : i64, loc_y = 2 : i64, size_x = 4 : i64, size_y = 4 : i64, sym_name = "herd_0"}
    }
  }
  air.channel @channel_0 [4, 1]
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
  func.func @attention_bf16(%arg0: memref<512x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<512x64xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = airrt.wait_all : !airrt.event
    affine.for %arg4 = 0 to 1 {
      %p = airrt.segment_load "attn_seg" : i64
      %p_0 = airrt.segment_load "attn_seg" : i64
      %c28672 = arith.constant 28672 : index
      %c20480 = arith.constant 20480 : index
      %c12288 = arith.constant 12288 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c4096 = arith.constant 4096 : index
      %c64 = arith.constant 64 : index
      %c8192 = arith.constant 8192 : index
      %c16384 = arith.constant 16384 : index
      %c24576 = arith.constant 24576 : index
      %c3 = arith.constant 3 : index
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_i64 = arith.constant 0 : i64
      %c0_3 = arith.constant 0 : index
      %c1_4 = arith.constant 1 : index
      %c33_i32 = arith.constant 33 : i32
      %1 = arith.index_cast %arg4 : index to i64
      %2 = arith.index_cast %c0_1 : index to i64
      %3 = arith.index_cast %c0_1 : index to i64
      %4 = arith.index_cast %c0_1 : index to i64
      %5 = arith.index_cast %c0_1 : index to i64
      %6 = arith.index_cast %c4096 : index to i64
      %7 = arith.index_cast %c8 : index to i64
      %8 = arith.index_cast %c64 : index to i64
      %9 = arith.index_cast %c1_2 : index to i64
      %10 = arith.index_cast %c4 : index to i64
      %11 = arith.index_cast %c8 : index to i64
      %12 = arith.index_cast %c64 : index to i64
      %13 = arith.index_cast %c8 : index to i64
      %14 = airrt.dma_memcpy_nd(%c33_i32, %1, %c0_i64, %arg0[%2, %3, %4, %5], [%10, %11, %12, %13], [%6, %7, %8, %9]) {chan_name = @QK2L1_0, metadata = @air_QK2L1_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %15 = airrt.wait_all %14 : !airrt.event
      %c0_i64_5 = arith.constant 0 : i64
      %c0_6 = arith.constant 0 : index
      %c1_7 = arith.constant 1 : index
      %c33_i32_8 = arith.constant 33 : i32
      %16 = arith.index_cast %arg4 : index to i64
      %17 = arith.index_cast %c0_1 : index to i64
      %18 = arith.index_cast %c0_1 : index to i64
      %19 = arith.index_cast %c0_1 : index to i64
      %20 = arith.index_cast %c0_1 : index to i64
      %21 = arith.index_cast %c4096 : index to i64
      %22 = arith.index_cast %c8 : index to i64
      %23 = arith.index_cast %c64 : index to i64
      %24 = arith.index_cast %c1_2 : index to i64
      %25 = arith.index_cast %c2 : index to i64
      %26 = arith.index_cast %c8 : index to i64
      %27 = arith.index_cast %c64 : index to i64
      %28 = arith.index_cast %c8 : index to i64
      %29 = airrt.dma_memcpy_nd(%c33_i32_8, %16, %c0_i64_5, %arg1[%17, %18, %19, %20], [%25, %26, %27, %28], [%21, %22, %23, %24]) {chan_name = @QK2L1_0, metadata = @air_QK2L1_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %30 = airrt.wait_all %29 : !airrt.event
      %c0_i64_9 = arith.constant 0 : i64
      %c0_10 = arith.constant 0 : index
      %c1_11 = arith.constant 1 : index
      %c34_i32 = arith.constant 34 : i32
      %31 = arith.index_cast %arg4 : index to i64
      %32 = arith.index_cast %c0_1 : index to i64
      %33 = arith.index_cast %c0_1 : index to i64
      %34 = arith.index_cast %c0_1 : index to i64
      %35 = arith.index_cast %c0_1 : index to i64
      %36 = arith.index_cast %c4096 : index to i64
      %37 = arith.index_cast %c8 : index to i64
      %38 = arith.index_cast %c64 : index to i64
      %39 = arith.index_cast %c1_2 : index to i64
      %40 = arith.index_cast %c4 : index to i64
      %41 = arith.index_cast %c8 : index to i64
      %42 = arith.index_cast %c64 : index to i64
      %43 = arith.index_cast %c8 : index to i64
      %44 = airrt.dma_memcpy_nd(%c34_i32, %31, %c0_i64_9, %arg0[%32, %33, %34, %35], [%40, %41, %42, %43], [%36, %37, %38, %39]) {chan_name = @QK2L1_1, metadata = @air_QK2L1_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %45 = airrt.wait_all %44 : !airrt.event
      %c0_i64_12 = arith.constant 0 : i64
      %c0_13 = arith.constant 0 : index
      %c1_14 = arith.constant 1 : index
      %c34_i32_15 = arith.constant 34 : i32
      %46 = arith.index_cast %arg4 : index to i64
      %47 = arith.index_cast %c0_1 : index to i64
      %48 = arith.index_cast %c0_1 : index to i64
      %49 = arith.index_cast %c0_1 : index to i64
      %50 = arith.index_cast %c8192 : index to i64
      %51 = arith.index_cast %c4096 : index to i64
      %52 = arith.index_cast %c8 : index to i64
      %53 = arith.index_cast %c64 : index to i64
      %54 = arith.index_cast %c1_2 : index to i64
      %55 = arith.index_cast %c2 : index to i64
      %56 = arith.index_cast %c8 : index to i64
      %57 = arith.index_cast %c64 : index to i64
      %58 = arith.index_cast %c8 : index to i64
      %59 = airrt.dma_memcpy_nd(%c34_i32_15, %46, %c0_i64_12, %arg1[%47, %48, %49, %50], [%55, %56, %57, %58], [%51, %52, %53, %54]) {chan_name = @QK2L1_1, metadata = @air_QK2L1_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %60 = airrt.wait_all %59 : !airrt.event
      %c0_i64_16 = arith.constant 0 : i64
      %c0_17 = arith.constant 0 : index
      %c1_18 = arith.constant 1 : index
      %c35_i32 = arith.constant 35 : i32
      %61 = arith.index_cast %arg4 : index to i64
      %62 = arith.index_cast %c0_1 : index to i64
      %63 = arith.index_cast %c0_1 : index to i64
      %64 = arith.index_cast %c0_1 : index to i64
      %65 = arith.index_cast %c0_1 : index to i64
      %66 = arith.index_cast %c4096 : index to i64
      %67 = arith.index_cast %c8 : index to i64
      %68 = arith.index_cast %c64 : index to i64
      %69 = arith.index_cast %c1_2 : index to i64
      %70 = arith.index_cast %c4 : index to i64
      %71 = arith.index_cast %c8 : index to i64
      %72 = arith.index_cast %c64 : index to i64
      %73 = arith.index_cast %c8 : index to i64
      %74 = airrt.dma_memcpy_nd(%c35_i32, %61, %c0_i64_16, %arg0[%62, %63, %64, %65], [%70, %71, %72, %73], [%66, %67, %68, %69]) {chan_name = @QK2L1_2, metadata = @air_QK2L1_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %75 = airrt.wait_all %74 : !airrt.event
      %c0_i64_19 = arith.constant 0 : i64
      %c0_20 = arith.constant 0 : index
      %c1_21 = arith.constant 1 : index
      %c35_i32_22 = arith.constant 35 : i32
      %76 = arith.index_cast %arg4 : index to i64
      %77 = arith.index_cast %c0_1 : index to i64
      %78 = arith.index_cast %c0_1 : index to i64
      %79 = arith.index_cast %c0_1 : index to i64
      %80 = arith.index_cast %c16384 : index to i64
      %81 = arith.index_cast %c4096 : index to i64
      %82 = arith.index_cast %c8 : index to i64
      %83 = arith.index_cast %c64 : index to i64
      %84 = arith.index_cast %c1_2 : index to i64
      %85 = arith.index_cast %c2 : index to i64
      %86 = arith.index_cast %c8 : index to i64
      %87 = arith.index_cast %c64 : index to i64
      %88 = arith.index_cast %c8 : index to i64
      %89 = airrt.dma_memcpy_nd(%c35_i32_22, %76, %c0_i64_19, %arg1[%77, %78, %79, %80], [%85, %86, %87, %88], [%81, %82, %83, %84]) {chan_name = @QK2L1_2, metadata = @air_QK2L1_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %90 = airrt.wait_all %89 : !airrt.event
      %c0_i64_23 = arith.constant 0 : i64
      %c0_24 = arith.constant 0 : index
      %c1_25 = arith.constant 1 : index
      %c36_i32 = arith.constant 36 : i32
      %91 = arith.index_cast %arg4 : index to i64
      %92 = arith.index_cast %c0_1 : index to i64
      %93 = arith.index_cast %c0_1 : index to i64
      %94 = arith.index_cast %c0_1 : index to i64
      %95 = arith.index_cast %c0_1 : index to i64
      %96 = arith.index_cast %c4096 : index to i64
      %97 = arith.index_cast %c8 : index to i64
      %98 = arith.index_cast %c64 : index to i64
      %99 = arith.index_cast %c1_2 : index to i64
      %100 = arith.index_cast %c4 : index to i64
      %101 = arith.index_cast %c8 : index to i64
      %102 = arith.index_cast %c64 : index to i64
      %103 = arith.index_cast %c8 : index to i64
      %104 = airrt.dma_memcpy_nd(%c36_i32, %91, %c0_i64_23, %arg0[%92, %93, %94, %95], [%100, %101, %102, %103], [%96, %97, %98, %99]) {chan_name = @QK2L1_3, metadata = @air_QK2L1_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %105 = airrt.wait_all %104 : !airrt.event
      %c0_i64_26 = arith.constant 0 : i64
      %c0_27 = arith.constant 0 : index
      %c1_28 = arith.constant 1 : index
      %c36_i32_29 = arith.constant 36 : i32
      %106 = arith.index_cast %arg4 : index to i64
      %107 = arith.index_cast %c0_1 : index to i64
      %108 = arith.index_cast %c0_1 : index to i64
      %109 = arith.index_cast %c0_1 : index to i64
      %110 = arith.index_cast %c24576 : index to i64
      %111 = arith.index_cast %c4096 : index to i64
      %112 = arith.index_cast %c8 : index to i64
      %113 = arith.index_cast %c64 : index to i64
      %114 = arith.index_cast %c1_2 : index to i64
      %115 = arith.index_cast %c2 : index to i64
      %116 = arith.index_cast %c8 : index to i64
      %117 = arith.index_cast %c64 : index to i64
      %118 = arith.index_cast %c8 : index to i64
      %119 = airrt.dma_memcpy_nd(%c36_i32_29, %106, %c0_i64_26, %arg1[%107, %108, %109, %110], [%115, %116, %117, %118], [%111, %112, %113, %114]) {chan_name = @QK2L1_3, metadata = @air_QK2L1_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %120 = airrt.wait_all %119 : !airrt.event
      %c0_i64_30 = arith.constant 0 : i64
      %c0_31 = arith.constant 0 : index
      %c1_32 = arith.constant 1 : index
      %c17_i32 = arith.constant 17 : i32
      %121 = arith.index_cast %arg4 : index to i64
      %122 = arith.index_cast %c0_31 : index to i64
      %123 = arith.index_cast %c0_31 : index to i64
      %124 = arith.index_cast %c0_31 : index to i64
      %125 = arith.index_cast %c0_1 : index to i64
      %126 = arith.index_cast %c0_31 : index to i64
      %127 = arith.index_cast %c0_31 : index to i64
      %128 = arith.index_cast %c0_31 : index to i64
      %129 = arith.index_cast %c1_2 : index to i64
      %130 = arith.index_cast %c1_32 : index to i64
      %131 = arith.index_cast %c1_32 : index to i64
      %132 = arith.index_cast %c1_32 : index to i64
      %133 = arith.index_cast %c8192 : index to i64
      %134 = airrt.dma_memcpy_nd(%c17_i32, %121, %c0_i64_30, %arg2[%122, %123, %124, %125], [%130, %131, %132, %133], [%126, %127, %128, %129]) {chan_name = @VIn_0, metadata = @air_VIn_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %135 = airrt.wait_all %134 : !airrt.event
      %c0_i64_33 = arith.constant 0 : i64
      %c0_34 = arith.constant 0 : index
      %c1_35 = arith.constant 1 : index
      %c19_i32 = arith.constant 19 : i32
      %136 = arith.index_cast %arg4 : index to i64
      %137 = arith.index_cast %c0_34 : index to i64
      %138 = arith.index_cast %c0_34 : index to i64
      %139 = arith.index_cast %c0_34 : index to i64
      %140 = arith.index_cast %c8192 : index to i64
      %141 = arith.index_cast %c0_34 : index to i64
      %142 = arith.index_cast %c0_34 : index to i64
      %143 = arith.index_cast %c0_34 : index to i64
      %144 = arith.index_cast %c1_2 : index to i64
      %145 = arith.index_cast %c1_35 : index to i64
      %146 = arith.index_cast %c1_35 : index to i64
      %147 = arith.index_cast %c1_35 : index to i64
      %148 = arith.index_cast %c8192 : index to i64
      %149 = airrt.dma_memcpy_nd(%c19_i32, %136, %c0_i64_33, %arg2[%137, %138, %139, %140], [%145, %146, %147, %148], [%141, %142, %143, %144]) {chan_name = @VIn_1, metadata = @air_VIn_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %150 = airrt.wait_all %149 : !airrt.event
      %c0_i64_36 = arith.constant 0 : i64
      %c0_37 = arith.constant 0 : index
      %c1_38 = arith.constant 1 : index
      %c21_i32 = arith.constant 21 : i32
      %151 = arith.index_cast %arg4 : index to i64
      %152 = arith.index_cast %c0_37 : index to i64
      %153 = arith.index_cast %c0_37 : index to i64
      %154 = arith.index_cast %c0_37 : index to i64
      %155 = arith.index_cast %c16384 : index to i64
      %156 = arith.index_cast %c0_37 : index to i64
      %157 = arith.index_cast %c0_37 : index to i64
      %158 = arith.index_cast %c0_37 : index to i64
      %159 = arith.index_cast %c1_2 : index to i64
      %160 = arith.index_cast %c1_38 : index to i64
      %161 = arith.index_cast %c1_38 : index to i64
      %162 = arith.index_cast %c1_38 : index to i64
      %163 = arith.index_cast %c8192 : index to i64
      %164 = airrt.dma_memcpy_nd(%c21_i32, %151, %c0_i64_36, %arg2[%152, %153, %154, %155], [%160, %161, %162, %163], [%156, %157, %158, %159]) {chan_name = @VIn_2, metadata = @air_VIn_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %165 = airrt.wait_all %164 : !airrt.event
      %c0_i64_39 = arith.constant 0 : i64
      %c0_40 = arith.constant 0 : index
      %c1_41 = arith.constant 1 : index
      %c23_i32 = arith.constant 23 : i32
      %166 = arith.index_cast %arg4 : index to i64
      %167 = arith.index_cast %c0_40 : index to i64
      %168 = arith.index_cast %c0_40 : index to i64
      %169 = arith.index_cast %c0_40 : index to i64
      %170 = arith.index_cast %c24576 : index to i64
      %171 = arith.index_cast %c0_40 : index to i64
      %172 = arith.index_cast %c0_40 : index to i64
      %173 = arith.index_cast %c0_40 : index to i64
      %174 = arith.index_cast %c1_2 : index to i64
      %175 = arith.index_cast %c1_41 : index to i64
      %176 = arith.index_cast %c1_41 : index to i64
      %177 = arith.index_cast %c1_41 : index to i64
      %178 = arith.index_cast %c8192 : index to i64
      %179 = airrt.dma_memcpy_nd(%c23_i32, %166, %c0_i64_39, %arg2[%167, %168, %169, %170], [%175, %176, %177, %178], [%171, %172, %173, %174]) {chan_name = @VIn_3, metadata = @air_VIn_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %180 = airrt.wait_all %179 : !airrt.event
      %c0_i64_42 = arith.constant 0 : i64
      %c0_43 = arith.constant 0 : index
      %c1_44 = arith.constant 1 : index
      %c29_i32 = arith.constant 29 : i32
      %181 = arith.index_cast %arg4 : index to i64
      %182 = arith.index_cast %c0_43 : index to i64
      %183 = arith.index_cast %c0_43 : index to i64
      %184 = arith.index_cast %c0_43 : index to i64
      %185 = arith.index_cast %c0_1 : index to i64
      %186 = arith.index_cast %c0_43 : index to i64
      %187 = arith.index_cast %c0_43 : index to i64
      %188 = arith.index_cast %c0_43 : index to i64
      %189 = arith.index_cast %c1_2 : index to i64
      %190 = arith.index_cast %c1_44 : index to i64
      %191 = arith.index_cast %c1_44 : index to i64
      %192 = arith.index_cast %c1_44 : index to i64
      %193 = arith.index_cast %c4096 : index to i64
      %194 = airrt.dma_memcpy_nd(%c29_i32, %181, %c0_i64_42, %arg3[%182, %183, %184, %185], [%190, %191, %192, %193], [%186, %187, %188, %189]) {chan_name = @channel_0, metadata = @air_channel_0_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %195 = airrt.wait_all %194 : !airrt.event
      %c0_i64_45 = arith.constant 0 : i64
      %c0_46 = arith.constant 0 : index
      %c1_47 = arith.constant 1 : index
      %c29_i32_48 = arith.constant 29 : i32
      %196 = arith.index_cast %arg4 : index to i64
      %197 = arith.index_cast %c0_46 : index to i64
      %198 = arith.index_cast %c0_46 : index to i64
      %199 = arith.index_cast %c0_46 : index to i64
      %200 = arith.index_cast %c4096 : index to i64
      %201 = arith.index_cast %c0_46 : index to i64
      %202 = arith.index_cast %c0_46 : index to i64
      %203 = arith.index_cast %c0_46 : index to i64
      %204 = arith.index_cast %c1_2 : index to i64
      %205 = arith.index_cast %c1_47 : index to i64
      %206 = arith.index_cast %c1_47 : index to i64
      %207 = arith.index_cast %c1_47 : index to i64
      %208 = arith.index_cast %c4096 : index to i64
      %209 = airrt.dma_memcpy_nd(%c29_i32_48, %196, %c0_i64_45, %arg3[%197, %198, %199, %200], [%205, %206, %207, %208], [%201, %202, %203, %204]) {chan_name = @channel_0, metadata = @air_channel_0_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %210 = airrt.wait_all %209 : !airrt.event
      %c0_i64_49 = arith.constant 0 : i64
      %c0_50 = arith.constant 0 : index
      %c1_51 = arith.constant 1 : index
      %c29_i32_52 = arith.constant 29 : i32
      %211 = arith.index_cast %arg4 : index to i64
      %212 = arith.index_cast %c0_50 : index to i64
      %213 = arith.index_cast %c0_50 : index to i64
      %214 = arith.index_cast %c0_50 : index to i64
      %215 = arith.index_cast %c8192 : index to i64
      %216 = arith.index_cast %c0_50 : index to i64
      %217 = arith.index_cast %c0_50 : index to i64
      %218 = arith.index_cast %c0_50 : index to i64
      %219 = arith.index_cast %c1_2 : index to i64
      %220 = arith.index_cast %c1_51 : index to i64
      %221 = arith.index_cast %c1_51 : index to i64
      %222 = arith.index_cast %c1_51 : index to i64
      %223 = arith.index_cast %c4096 : index to i64
      %224 = airrt.dma_memcpy_nd(%c29_i32_52, %211, %c0_i64_49, %arg3[%212, %213, %214, %215], [%220, %221, %222, %223], [%216, %217, %218, %219]) {chan_name = @channel_0, metadata = @air_channel_0_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %225 = airrt.wait_all %224 : !airrt.event
      %c0_i64_53 = arith.constant 0 : i64
      %c0_54 = arith.constant 0 : index
      %c1_55 = arith.constant 1 : index
      %c29_i32_56 = arith.constant 29 : i32
      %226 = arith.index_cast %arg4 : index to i64
      %227 = arith.index_cast %c0_54 : index to i64
      %228 = arith.index_cast %c0_54 : index to i64
      %229 = arith.index_cast %c0_54 : index to i64
      %230 = arith.index_cast %c12288 : index to i64
      %231 = arith.index_cast %c0_54 : index to i64
      %232 = arith.index_cast %c0_54 : index to i64
      %233 = arith.index_cast %c0_54 : index to i64
      %234 = arith.index_cast %c1_2 : index to i64
      %235 = arith.index_cast %c1_55 : index to i64
      %236 = arith.index_cast %c1_55 : index to i64
      %237 = arith.index_cast %c1_55 : index to i64
      %238 = arith.index_cast %c4096 : index to i64
      %239 = airrt.dma_memcpy_nd(%c29_i32_56, %226, %c0_i64_53, %arg3[%227, %228, %229, %230], [%235, %236, %237, %238], [%231, %232, %233, %234]) {chan_name = @channel_0, metadata = @air_channel_0_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %240 = airrt.wait_all %239 : !airrt.event
      %c0_57 = arith.constant 0 : index
      %c1_58 = arith.constant 1 : index
      %241 = airrt.wait_all : !airrt.event
      affine.for %arg5 = 0 to 1 {
        affine.for %arg6 = 0 to 1 {
          %c3_125 = arith.constant 3 : index
          %c64_126 = arith.constant 64 : index
          %c8_127 = arith.constant 8 : index
          %c1_128 = arith.constant 1 : index
          %c2_129 = arith.constant 2 : index
          %c0_130 = arith.constant 0 : index
          %c4_131 = arith.constant 4 : index
          %483 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %484 = airrt.wait_all : !airrt.event
          %485 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %486 = airrt.wait_all : !airrt.event
          %487 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %488 = airrt.wait_all : !airrt.event
          %489 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %490 = airrt.wait_all : !airrt.event
          %491 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %492 = airrt.wait_all : !airrt.event
          %493 = airrt.wait_all %492 : !airrt.event
          %494 = airrt.wait_all %493 : !airrt.event
          %495 = airrt.wait_all %492, %494 : !airrt.event
          %496 = airrt.wait_all %495 : !airrt.event
          airrt.dealloc %491 : memref<64x64xbf16, 1 : i32>
          %497 = airrt.wait_all : !airrt.event
          %498 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %499 = airrt.wait_all : !airrt.event
          %500 = airrt.wait_all %499 : !airrt.event
          %501 = airrt.wait_all %500 : !airrt.event
          %502 = airrt.wait_all %499, %501 : !airrt.event
          %503 = airrt.wait_all %502 : !airrt.event
          airrt.dealloc %498 : memref<64x64xbf16, 1 : i32>
          %504 = airrt.wait_all : !airrt.event
          %505 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %506 = airrt.wait_all : !airrt.event
          %507 = airrt.wait_all %506 : !airrt.event
          %508 = airrt.wait_all %507 : !airrt.event
          %509 = airrt.wait_all %506, %508 : !airrt.event
          %510 = airrt.wait_all %509 : !airrt.event
          airrt.dealloc %505 : memref<64x64xbf16, 1 : i32>
          %511 = airrt.wait_all : !airrt.event
          %512 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %513 = airrt.wait_all : !airrt.event
          %514 = airrt.wait_all %513 : !airrt.event
          %515 = airrt.wait_all %514 : !airrt.event
          %516 = airrt.wait_all %513, %515 : !airrt.event
          %517 = airrt.wait_all %516 : !airrt.event
          airrt.dealloc %512 : memref<64x64xbf16, 1 : i32>
          %518 = airrt.wait_all : !airrt.event
          %519 = airrt.wait_all %484 : !airrt.event
          %520 = airrt.wait_all %486 : !airrt.event
          %521 = airrt.wait_all %488 : !airrt.event
          %522 = airrt.wait_all %490 : !airrt.event
          %523 = airrt.wait_all %519 : !airrt.event
          %524 = airrt.wait_all %520 : !airrt.event
          %525 = airrt.wait_all %521 : !airrt.event
          %526 = airrt.wait_all %522 : !airrt.event
          %h = airrt.herd_load "herd_0" () {segment_name = "attn_seg"} : () -> i64
          %527 = airrt.wait_all : !airrt.event
          airrt.dealloc %489 : memref<64x64xbf16, 1 : i32>
          %528 = airrt.wait_all : !airrt.event
          airrt.dealloc %487 : memref<64x64xbf16, 1 : i32>
          %529 = airrt.wait_all : !airrt.event
          airrt.dealloc %485 : memref<64x64xbf16, 1 : i32>
          %530 = airrt.wait_all : !airrt.event
          airrt.dealloc %483 : memref<64x64xbf16, 1 : i32>
          %531 = airrt.wait_all : !airrt.event
          airrt.wait_all %496, %503, %510, %517, %527, %528, %529, %530, %531 {air.segment_end}
        }
      }
      airrt.wait_all %150, %180, %210, %240, %30, %15, %45, %60, %90, %75, %105, %120, %241, %225, %195, %165, %135 {air.launch_end}
      %c0_i64_59 = arith.constant 0 : i64
      %c0_60 = arith.constant 0 : index
      %c1_61 = arith.constant 1 : index
      %c33_i32_62 = arith.constant 33 : i32
      %242 = arith.index_cast %arg4 : index to i64
      %243 = arith.index_cast %c0_1 : index to i64
      %244 = arith.index_cast %c0_1 : index to i64
      %245 = arith.index_cast %c0_1 : index to i64
      %246 = arith.index_cast %c16384 : index to i64
      %247 = arith.index_cast %c4096 : index to i64
      %248 = arith.index_cast %c8 : index to i64
      %249 = arith.index_cast %c64 : index to i64
      %250 = arith.index_cast %c1_2 : index to i64
      %251 = arith.index_cast %c4 : index to i64
      %252 = arith.index_cast %c8 : index to i64
      %253 = arith.index_cast %c64 : index to i64
      %254 = arith.index_cast %c8 : index to i64
      %255 = airrt.dma_memcpy_nd(%c33_i32_62, %242, %c0_i64_59, %arg0[%243, %244, %245, %246], [%251, %252, %253, %254], [%247, %248, %249, %250]) {chan_name = @QK2L1_0, metadata = @air_QK2L1_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %256 = airrt.wait_all %255 : !airrt.event
      %c0_i64_63 = arith.constant 0 : i64
      %c0_64 = arith.constant 0 : index
      %c1_65 = arith.constant 1 : index
      %c33_i32_66 = arith.constant 33 : i32
      %257 = arith.index_cast %arg4 : index to i64
      %258 = arith.index_cast %c0_1 : index to i64
      %259 = arith.index_cast %c0_1 : index to i64
      %260 = arith.index_cast %c0_1 : index to i64
      %261 = arith.index_cast %c0_1 : index to i64
      %262 = arith.index_cast %c4096 : index to i64
      %263 = arith.index_cast %c8 : index to i64
      %264 = arith.index_cast %c64 : index to i64
      %265 = arith.index_cast %c1_2 : index to i64
      %266 = arith.index_cast %c2 : index to i64
      %267 = arith.index_cast %c8 : index to i64
      %268 = arith.index_cast %c64 : index to i64
      %269 = arith.index_cast %c8 : index to i64
      %270 = airrt.dma_memcpy_nd(%c33_i32_66, %257, %c0_i64_63, %arg1[%258, %259, %260, %261], [%266, %267, %268, %269], [%262, %263, %264, %265]) {chan_name = @QK2L1_0, metadata = @air_QK2L1_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %271 = airrt.wait_all %270 : !airrt.event
      %c0_i64_67 = arith.constant 0 : i64
      %c0_68 = arith.constant 0 : index
      %c1_69 = arith.constant 1 : index
      %c34_i32_70 = arith.constant 34 : i32
      %272 = arith.index_cast %arg4 : index to i64
      %273 = arith.index_cast %c0_1 : index to i64
      %274 = arith.index_cast %c0_1 : index to i64
      %275 = arith.index_cast %c0_1 : index to i64
      %276 = arith.index_cast %c16384 : index to i64
      %277 = arith.index_cast %c4096 : index to i64
      %278 = arith.index_cast %c8 : index to i64
      %279 = arith.index_cast %c64 : index to i64
      %280 = arith.index_cast %c1_2 : index to i64
      %281 = arith.index_cast %c4 : index to i64
      %282 = arith.index_cast %c8 : index to i64
      %283 = arith.index_cast %c64 : index to i64
      %284 = arith.index_cast %c8 : index to i64
      %285 = airrt.dma_memcpy_nd(%c34_i32_70, %272, %c0_i64_67, %arg0[%273, %274, %275, %276], [%281, %282, %283, %284], [%277, %278, %279, %280]) {chan_name = @QK2L1_1, metadata = @air_QK2L1_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %286 = airrt.wait_all %285 : !airrt.event
      %c0_i64_71 = arith.constant 0 : i64
      %c0_72 = arith.constant 0 : index
      %c1_73 = arith.constant 1 : index
      %c34_i32_74 = arith.constant 34 : i32
      %287 = arith.index_cast %arg4 : index to i64
      %288 = arith.index_cast %c0_1 : index to i64
      %289 = arith.index_cast %c0_1 : index to i64
      %290 = arith.index_cast %c0_1 : index to i64
      %291 = arith.index_cast %c8192 : index to i64
      %292 = arith.index_cast %c4096 : index to i64
      %293 = arith.index_cast %c8 : index to i64
      %294 = arith.index_cast %c64 : index to i64
      %295 = arith.index_cast %c1_2 : index to i64
      %296 = arith.index_cast %c2 : index to i64
      %297 = arith.index_cast %c8 : index to i64
      %298 = arith.index_cast %c64 : index to i64
      %299 = arith.index_cast %c8 : index to i64
      %300 = airrt.dma_memcpy_nd(%c34_i32_74, %287, %c0_i64_71, %arg1[%288, %289, %290, %291], [%296, %297, %298, %299], [%292, %293, %294, %295]) {chan_name = @QK2L1_1, metadata = @air_QK2L1_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %301 = airrt.wait_all %300 : !airrt.event
      %c0_i64_75 = arith.constant 0 : i64
      %c0_76 = arith.constant 0 : index
      %c1_77 = arith.constant 1 : index
      %c35_i32_78 = arith.constant 35 : i32
      %302 = arith.index_cast %arg4 : index to i64
      %303 = arith.index_cast %c0_1 : index to i64
      %304 = arith.index_cast %c0_1 : index to i64
      %305 = arith.index_cast %c0_1 : index to i64
      %306 = arith.index_cast %c16384 : index to i64
      %307 = arith.index_cast %c4096 : index to i64
      %308 = arith.index_cast %c8 : index to i64
      %309 = arith.index_cast %c64 : index to i64
      %310 = arith.index_cast %c1_2 : index to i64
      %311 = arith.index_cast %c4 : index to i64
      %312 = arith.index_cast %c8 : index to i64
      %313 = arith.index_cast %c64 : index to i64
      %314 = arith.index_cast %c8 : index to i64
      %315 = airrt.dma_memcpy_nd(%c35_i32_78, %302, %c0_i64_75, %arg0[%303, %304, %305, %306], [%311, %312, %313, %314], [%307, %308, %309, %310]) {chan_name = @QK2L1_2, metadata = @air_QK2L1_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %316 = airrt.wait_all %315 : !airrt.event
      %c0_i64_79 = arith.constant 0 : i64
      %c0_80 = arith.constant 0 : index
      %c1_81 = arith.constant 1 : index
      %c35_i32_82 = arith.constant 35 : i32
      %317 = arith.index_cast %arg4 : index to i64
      %318 = arith.index_cast %c0_1 : index to i64
      %319 = arith.index_cast %c0_1 : index to i64
      %320 = arith.index_cast %c0_1 : index to i64
      %321 = arith.index_cast %c16384 : index to i64
      %322 = arith.index_cast %c4096 : index to i64
      %323 = arith.index_cast %c8 : index to i64
      %324 = arith.index_cast %c64 : index to i64
      %325 = arith.index_cast %c1_2 : index to i64
      %326 = arith.index_cast %c2 : index to i64
      %327 = arith.index_cast %c8 : index to i64
      %328 = arith.index_cast %c64 : index to i64
      %329 = arith.index_cast %c8 : index to i64
      %330 = airrt.dma_memcpy_nd(%c35_i32_82, %317, %c0_i64_79, %arg1[%318, %319, %320, %321], [%326, %327, %328, %329], [%322, %323, %324, %325]) {chan_name = @QK2L1_2, metadata = @air_QK2L1_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %331 = airrt.wait_all %330 : !airrt.event
      %c0_i64_83 = arith.constant 0 : i64
      %c0_84 = arith.constant 0 : index
      %c1_85 = arith.constant 1 : index
      %c36_i32_86 = arith.constant 36 : i32
      %332 = arith.index_cast %arg4 : index to i64
      %333 = arith.index_cast %c0_1 : index to i64
      %334 = arith.index_cast %c0_1 : index to i64
      %335 = arith.index_cast %c0_1 : index to i64
      %336 = arith.index_cast %c16384 : index to i64
      %337 = arith.index_cast %c4096 : index to i64
      %338 = arith.index_cast %c8 : index to i64
      %339 = arith.index_cast %c64 : index to i64
      %340 = arith.index_cast %c1_2 : index to i64
      %341 = arith.index_cast %c4 : index to i64
      %342 = arith.index_cast %c8 : index to i64
      %343 = arith.index_cast %c64 : index to i64
      %344 = arith.index_cast %c8 : index to i64
      %345 = airrt.dma_memcpy_nd(%c36_i32_86, %332, %c0_i64_83, %arg0[%333, %334, %335, %336], [%341, %342, %343, %344], [%337, %338, %339, %340]) {chan_name = @QK2L1_3, metadata = @air_QK2L1_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %346 = airrt.wait_all %345 : !airrt.event
      %c0_i64_87 = arith.constant 0 : i64
      %c0_88 = arith.constant 0 : index
      %c1_89 = arith.constant 1 : index
      %c36_i32_90 = arith.constant 36 : i32
      %347 = arith.index_cast %arg4 : index to i64
      %348 = arith.index_cast %c0_1 : index to i64
      %349 = arith.index_cast %c0_1 : index to i64
      %350 = arith.index_cast %c0_1 : index to i64
      %351 = arith.index_cast %c24576 : index to i64
      %352 = arith.index_cast %c4096 : index to i64
      %353 = arith.index_cast %c8 : index to i64
      %354 = arith.index_cast %c64 : index to i64
      %355 = arith.index_cast %c1_2 : index to i64
      %356 = arith.index_cast %c2 : index to i64
      %357 = arith.index_cast %c8 : index to i64
      %358 = arith.index_cast %c64 : index to i64
      %359 = arith.index_cast %c8 : index to i64
      %360 = airrt.dma_memcpy_nd(%c36_i32_90, %347, %c0_i64_87, %arg1[%348, %349, %350, %351], [%356, %357, %358, %359], [%352, %353, %354, %355]) {chan_name = @QK2L1_3, metadata = @air_QK2L1_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %361 = airrt.wait_all %360 : !airrt.event
      %c0_i64_91 = arith.constant 0 : i64
      %c0_92 = arith.constant 0 : index
      %c1_93 = arith.constant 1 : index
      %c17_i32_94 = arith.constant 17 : i32
      %362 = arith.index_cast %arg4 : index to i64
      %363 = arith.index_cast %c0_92 : index to i64
      %364 = arith.index_cast %c0_92 : index to i64
      %365 = arith.index_cast %c0_92 : index to i64
      %366 = arith.index_cast %c0_1 : index to i64
      %367 = arith.index_cast %c0_92 : index to i64
      %368 = arith.index_cast %c0_92 : index to i64
      %369 = arith.index_cast %c0_92 : index to i64
      %370 = arith.index_cast %c1_2 : index to i64
      %371 = arith.index_cast %c1_93 : index to i64
      %372 = arith.index_cast %c1_93 : index to i64
      %373 = arith.index_cast %c1_93 : index to i64
      %374 = arith.index_cast %c8192 : index to i64
      %375 = airrt.dma_memcpy_nd(%c17_i32_94, %362, %c0_i64_91, %arg2[%363, %364, %365, %366], [%371, %372, %373, %374], [%367, %368, %369, %370]) {chan_name = @VIn_0, metadata = @air_VIn_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %376 = airrt.wait_all %375 : !airrt.event
      %c0_i64_95 = arith.constant 0 : i64
      %c0_96 = arith.constant 0 : index
      %c1_97 = arith.constant 1 : index
      %c19_i32_98 = arith.constant 19 : i32
      %377 = arith.index_cast %arg4 : index to i64
      %378 = arith.index_cast %c0_96 : index to i64
      %379 = arith.index_cast %c0_96 : index to i64
      %380 = arith.index_cast %c0_96 : index to i64
      %381 = arith.index_cast %c8192 : index to i64
      %382 = arith.index_cast %c0_96 : index to i64
      %383 = arith.index_cast %c0_96 : index to i64
      %384 = arith.index_cast %c0_96 : index to i64
      %385 = arith.index_cast %c1_2 : index to i64
      %386 = arith.index_cast %c1_97 : index to i64
      %387 = arith.index_cast %c1_97 : index to i64
      %388 = arith.index_cast %c1_97 : index to i64
      %389 = arith.index_cast %c8192 : index to i64
      %390 = airrt.dma_memcpy_nd(%c19_i32_98, %377, %c0_i64_95, %arg2[%378, %379, %380, %381], [%386, %387, %388, %389], [%382, %383, %384, %385]) {chan_name = @VIn_1, metadata = @air_VIn_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %391 = airrt.wait_all %390 : !airrt.event
      %c0_i64_99 = arith.constant 0 : i64
      %c0_100 = arith.constant 0 : index
      %c1_101 = arith.constant 1 : index
      %c21_i32_102 = arith.constant 21 : i32
      %392 = arith.index_cast %arg4 : index to i64
      %393 = arith.index_cast %c0_100 : index to i64
      %394 = arith.index_cast %c0_100 : index to i64
      %395 = arith.index_cast %c0_100 : index to i64
      %396 = arith.index_cast %c16384 : index to i64
      %397 = arith.index_cast %c0_100 : index to i64
      %398 = arith.index_cast %c0_100 : index to i64
      %399 = arith.index_cast %c0_100 : index to i64
      %400 = arith.index_cast %c1_2 : index to i64
      %401 = arith.index_cast %c1_101 : index to i64
      %402 = arith.index_cast %c1_101 : index to i64
      %403 = arith.index_cast %c1_101 : index to i64
      %404 = arith.index_cast %c8192 : index to i64
      %405 = airrt.dma_memcpy_nd(%c21_i32_102, %392, %c0_i64_99, %arg2[%393, %394, %395, %396], [%401, %402, %403, %404], [%397, %398, %399, %400]) {chan_name = @VIn_2, metadata = @air_VIn_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %406 = airrt.wait_all %405 : !airrt.event
      %c0_i64_103 = arith.constant 0 : i64
      %c0_104 = arith.constant 0 : index
      %c1_105 = arith.constant 1 : index
      %c23_i32_106 = arith.constant 23 : i32
      %407 = arith.index_cast %arg4 : index to i64
      %408 = arith.index_cast %c0_104 : index to i64
      %409 = arith.index_cast %c0_104 : index to i64
      %410 = arith.index_cast %c0_104 : index to i64
      %411 = arith.index_cast %c24576 : index to i64
      %412 = arith.index_cast %c0_104 : index to i64
      %413 = arith.index_cast %c0_104 : index to i64
      %414 = arith.index_cast %c0_104 : index to i64
      %415 = arith.index_cast %c1_2 : index to i64
      %416 = arith.index_cast %c1_105 : index to i64
      %417 = arith.index_cast %c1_105 : index to i64
      %418 = arith.index_cast %c1_105 : index to i64
      %419 = arith.index_cast %c8192 : index to i64
      %420 = airrt.dma_memcpy_nd(%c23_i32_106, %407, %c0_i64_103, %arg2[%408, %409, %410, %411], [%416, %417, %418, %419], [%412, %413, %414, %415]) {chan_name = @VIn_3, metadata = @air_VIn_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %421 = airrt.wait_all %420 : !airrt.event
      %c0_i64_107 = arith.constant 0 : i64
      %c0_108 = arith.constant 0 : index
      %c1_109 = arith.constant 1 : index
      %c29_i32_110 = arith.constant 29 : i32
      %422 = arith.index_cast %arg4 : index to i64
      %423 = arith.index_cast %c0_108 : index to i64
      %424 = arith.index_cast %c0_108 : index to i64
      %425 = arith.index_cast %c0_108 : index to i64
      %426 = arith.index_cast %c16384 : index to i64
      %427 = arith.index_cast %c0_108 : index to i64
      %428 = arith.index_cast %c0_108 : index to i64
      %429 = arith.index_cast %c0_108 : index to i64
      %430 = arith.index_cast %c1_2 : index to i64
      %431 = arith.index_cast %c1_109 : index to i64
      %432 = arith.index_cast %c1_109 : index to i64
      %433 = arith.index_cast %c1_109 : index to i64
      %434 = arith.index_cast %c4096 : index to i64
      %435 = airrt.dma_memcpy_nd(%c29_i32_110, %422, %c0_i64_107, %arg3[%423, %424, %425, %426], [%431, %432, %433, %434], [%427, %428, %429, %430]) {chan_name = @channel_0, metadata = @air_channel_0_0} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %436 = airrt.wait_all %435 : !airrt.event
      %c0_i64_111 = arith.constant 0 : i64
      %c0_112 = arith.constant 0 : index
      %c1_113 = arith.constant 1 : index
      %c29_i32_114 = arith.constant 29 : i32
      %437 = arith.index_cast %arg4 : index to i64
      %438 = arith.index_cast %c0_112 : index to i64
      %439 = arith.index_cast %c0_112 : index to i64
      %440 = arith.index_cast %c0_112 : index to i64
      %441 = arith.index_cast %c20480 : index to i64
      %442 = arith.index_cast %c0_112 : index to i64
      %443 = arith.index_cast %c0_112 : index to i64
      %444 = arith.index_cast %c0_112 : index to i64
      %445 = arith.index_cast %c1_2 : index to i64
      %446 = arith.index_cast %c1_113 : index to i64
      %447 = arith.index_cast %c1_113 : index to i64
      %448 = arith.index_cast %c1_113 : index to i64
      %449 = arith.index_cast %c4096 : index to i64
      %450 = airrt.dma_memcpy_nd(%c29_i32_114, %437, %c0_i64_111, %arg3[%438, %439, %440, %441], [%446, %447, %448, %449], [%442, %443, %444, %445]) {chan_name = @channel_0, metadata = @air_channel_0_1} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %451 = airrt.wait_all %450 : !airrt.event
      %c0_i64_115 = arith.constant 0 : i64
      %c0_116 = arith.constant 0 : index
      %c1_117 = arith.constant 1 : index
      %c29_i32_118 = arith.constant 29 : i32
      %452 = arith.index_cast %arg4 : index to i64
      %453 = arith.index_cast %c0_116 : index to i64
      %454 = arith.index_cast %c0_116 : index to i64
      %455 = arith.index_cast %c0_116 : index to i64
      %456 = arith.index_cast %c24576 : index to i64
      %457 = arith.index_cast %c0_116 : index to i64
      %458 = arith.index_cast %c0_116 : index to i64
      %459 = arith.index_cast %c0_116 : index to i64
      %460 = arith.index_cast %c1_2 : index to i64
      %461 = arith.index_cast %c1_117 : index to i64
      %462 = arith.index_cast %c1_117 : index to i64
      %463 = arith.index_cast %c1_117 : index to i64
      %464 = arith.index_cast %c4096 : index to i64
      %465 = airrt.dma_memcpy_nd(%c29_i32_118, %452, %c0_i64_115, %arg3[%453, %454, %455, %456], [%461, %462, %463, %464], [%457, %458, %459, %460]) {chan_name = @channel_0, metadata = @air_channel_0_2} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %466 = airrt.wait_all %465 : !airrt.event
      %c0_i64_119 = arith.constant 0 : i64
      %c0_120 = arith.constant 0 : index
      %c1_121 = arith.constant 1 : index
      %c29_i32_122 = arith.constant 29 : i32
      %467 = arith.index_cast %arg4 : index to i64
      %468 = arith.index_cast %c0_120 : index to i64
      %469 = arith.index_cast %c0_120 : index to i64
      %470 = arith.index_cast %c0_120 : index to i64
      %471 = arith.index_cast %c28672 : index to i64
      %472 = arith.index_cast %c0_120 : index to i64
      %473 = arith.index_cast %c0_120 : index to i64
      %474 = arith.index_cast %c0_120 : index to i64
      %475 = arith.index_cast %c1_2 : index to i64
      %476 = arith.index_cast %c1_121 : index to i64
      %477 = arith.index_cast %c1_121 : index to i64
      %478 = arith.index_cast %c1_121 : index to i64
      %479 = arith.index_cast %c4096 : index to i64
      %480 = airrt.dma_memcpy_nd(%c29_i32_122, %467, %c0_i64_119, %arg3[%468, %469, %470, %471], [%476, %477, %478, %479], [%472, %473, %474, %475]) {chan_name = @channel_0, metadata = @air_channel_0_3} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      %481 = airrt.wait_all %480 : !airrt.event
      %c0_123 = arith.constant 0 : index
      %c1_124 = arith.constant 1 : index
      %482 = airrt.wait_all : !airrt.event
      affine.for %arg5 = 0 to 1 {
        affine.for %arg6 = 0 to 1 {
          %c3_125 = arith.constant 3 : index
          %c64_126 = arith.constant 64 : index
          %c8_127 = arith.constant 8 : index
          %c1_128 = arith.constant 1 : index
          %c2_129 = arith.constant 2 : index
          %c0_130 = arith.constant 0 : index
          %c4_131 = arith.constant 4 : index
          %483 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %484 = airrt.wait_all : !airrt.event
          %485 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %486 = airrt.wait_all : !airrt.event
          %487 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %488 = airrt.wait_all : !airrt.event
          %489 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %490 = airrt.wait_all : !airrt.event
          %491 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %492 = airrt.wait_all : !airrt.event
          %493 = airrt.wait_all %492 : !airrt.event
          %494 = airrt.wait_all %493 : !airrt.event
          %495 = airrt.wait_all %492, %494 : !airrt.event
          %496 = airrt.wait_all %495 : !airrt.event
          airrt.dealloc %491 : memref<64x64xbf16, 1 : i32>
          %497 = airrt.wait_all : !airrt.event
          %498 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %499 = airrt.wait_all : !airrt.event
          %500 = airrt.wait_all %499 : !airrt.event
          %501 = airrt.wait_all %500 : !airrt.event
          %502 = airrt.wait_all %499, %501 : !airrt.event
          %503 = airrt.wait_all %502 : !airrt.event
          airrt.dealloc %498 : memref<64x64xbf16, 1 : i32>
          %504 = airrt.wait_all : !airrt.event
          %505 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %506 = airrt.wait_all : !airrt.event
          %507 = airrt.wait_all %506 : !airrt.event
          %508 = airrt.wait_all %507 : !airrt.event
          %509 = airrt.wait_all %506, %508 : !airrt.event
          %510 = airrt.wait_all %509 : !airrt.event
          airrt.dealloc %505 : memref<64x64xbf16, 1 : i32>
          %511 = airrt.wait_all : !airrt.event
          %512 = airrt.alloc : memref<64x64xbf16, 1 : i32>
          %513 = airrt.wait_all : !airrt.event
          %514 = airrt.wait_all %513 : !airrt.event
          %515 = airrt.wait_all %514 : !airrt.event
          %516 = airrt.wait_all %513, %515 : !airrt.event
          %517 = airrt.wait_all %516 : !airrt.event
          airrt.dealloc %512 : memref<64x64xbf16, 1 : i32>
          %518 = airrt.wait_all : !airrt.event
          %519 = airrt.wait_all %484 : !airrt.event
          %520 = airrt.wait_all %486 : !airrt.event
          %521 = airrt.wait_all %488 : !airrt.event
          %522 = airrt.wait_all %490 : !airrt.event
          %523 = airrt.wait_all %519 : !airrt.event
          %524 = airrt.wait_all %520 : !airrt.event
          %525 = airrt.wait_all %521 : !airrt.event
          %526 = airrt.wait_all %522 : !airrt.event
          %h = airrt.herd_load "herd_0" () {segment_name = "attn_seg"} : () -> i64
          %527 = airrt.wait_all : !airrt.event
          airrt.dealloc %489 : memref<64x64xbf16, 1 : i32>
          %528 = airrt.wait_all : !airrt.event
          airrt.dealloc %487 : memref<64x64xbf16, 1 : i32>
          %529 = airrt.wait_all : !airrt.event
          airrt.dealloc %485 : memref<64x64xbf16, 1 : i32>
          %530 = airrt.wait_all : !airrt.event
          airrt.dealloc %483 : memref<64x64xbf16, 1 : i32>
          %531 = airrt.wait_all : !airrt.event
          airrt.wait_all %496, %503, %510, %517, %527, %528, %529, %530, %531 {air.segment_end}
        }
      }
      airrt.wait_all %391, %421, %451, %481, %271, %256, %286, %301, %331, %316, %346, %361, %482, %466, %436, %406, %376 {air.launch_end}
    } {affine_opt_label = "tiling"}
    return
  }
}
