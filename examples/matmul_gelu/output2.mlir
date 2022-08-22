#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c16 = arith.constant 16 : index
    %c384 = arith.constant 384 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %asyncToken, %valOut = air.region async  {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.region_terminator %2 : memref<24576x1024xbf16>
    } {id = 1 : i32} : (memref<24576x1024xbf16>)
    %asyncToken_0 = air.region async [%asyncToken]  : (!air.async.token) {
      linalg.fill ins(%cst : bf16) outs(%valOut : memref<24576x1024xbf16>)
      air.region_terminator
    } {id = 2 : i32}
    %asyncToken_1, %valOut_2 = air.region async  {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.region_terminator %2 : memref<24576x1024xbf16>
    } {id = 3 : i32} : (memref<24576x1024xbf16>)
    %asyncToken_3 = air.region async [%asyncToken_1, %asyncToken_0]  : (!air.async.token, !air.async.token) {
      memref.copy %valOut, %valOut_2 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
      air.region_terminator
    } {id = 4 : i32}
    %0 = air.launch async [%asyncToken_3] (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%valOut_2) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> attributes {id = 3 : i32} {
      %2 = air.partition async  args(%arg9=%arg2, %arg10=%arg3, %arg11=%arg4, %arg12=%arg5, %arg13=%arg6, %arg14=%arg7, %arg15=%arg8) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> attributes {id = 2 : i32} {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %asyncToken_6, %valOut_7 = air.region async  {
          %7 = affine.apply #map0()[%arg9]
          air.region_terminator %7 : index
        } {id = 5 : i32} : (index)
        %asyncToken_8, %valOut_9 = air.region async  {
          %7 = affine.apply #map0()[%arg10]
          air.region_terminator %7 : index
        } {id = 6 : i32} : (index)
        %3 = air.wait_all async [%asyncToken_6, %asyncToken_8] 
        %asyncToken_10, %valOut_11 = air.region async  {
          %7 = memref.alloc() : memref<64x64xbf16, 1>
          air.region_terminator %7 : memref<64x64xbf16, 1>
        } {id = 9 : i32} : (memref<64x64xbf16, 1>)
        %4 = air.dma_memcpy_nd async [%3, %asyncToken_10] (%valOut_11[] [] [], %arg15[%valOut_7, %valOut_9] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        %5 = scf.for %arg16 = %c0 to %c1024 step %c64 iter_args(%arg17 = %4) -> (!air.async.token) {
          %asyncToken_13, %valOut_14 = air.region async [%arg17]  : (!air.async.token) {
            %11 = memref.alloc() : memref<64x64xbf16, 1>
            air.region_terminator %11 : memref<64x64xbf16, 1>
          } {id = 7 : i32} : (memref<64x64xbf16, 1>)
          %asyncToken_15, %valOut_16 = air.region async [%arg17]  : (!air.async.token) {
            %11 = memref.alloc() : memref<64x64xbf16, 1>
            air.region_terminator %11 : memref<64x64xbf16, 1>
          } {id = 8 : i32} : (memref<64x64xbf16, 1>)
          %7 = air.dma_memcpy_nd async [%asyncToken_13, %arg17] (%valOut_14[] [] [], %arg13[%valOut_7, %arg16] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          %8 = air.dma_memcpy_nd async [%asyncToken_15, %arg17] (%valOut_16[] [] [], %arg14[%arg16, %valOut_9] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          %9 = air.herd @herd_0 async [%8, %arg17, %7]  tile (%arg18, %arg19) in (%arg20=%c2, %arg21=%c2) args(%arg22=%valOut_14, %arg23=%valOut_16, %arg24=%valOut_11) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 1 : i32} {
            %c1_19 = arith.constant 1 : index
            %c0_20 = arith.constant 0 : index
            %c64_21 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %asyncToken_22, %valOut_23 = air.region async  {
              %15 = affine.apply #map1()[%arg18]
              air.region_terminator %15 : index
            } {id = 10 : i32} : (index)
            %asyncToken_24, %valOut_25 = air.region async  {
              %15 = affine.apply #map1()[%arg19]
              air.region_terminator %15 : index
            } {id = 11 : i32} : (index)
            %11 = air.wait_all async [%asyncToken_22, %asyncToken_24] 
            %asyncToken_26, %valOut_27 = air.region async  {
              %15 = memref.alloc() : memref<32x32xbf16, 2>
              air.region_terminator %15 : memref<32x32xbf16, 2>
            } {id = 14 : i32} : (memref<32x32xbf16, 2>)
            %12 = air.dma_memcpy_nd async [%11, %asyncToken_26] (%valOut_27[] [] [], %arg24[%valOut_23, %valOut_25] [%c32, %c32] [%c64_21, %c1_19]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            %13 = scf.for %arg25 = %c0_20 to %c64_21 step %c32 iter_args(%arg26 = %12) -> (!air.async.token) {
              %asyncToken_29, %valOut_30 = air.region async [%arg26]  : (!air.async.token) {
                %18 = memref.alloc() : memref<32x32xbf16, 2>
                air.region_terminator %18 : memref<32x32xbf16, 2>
              } {id = 12 : i32} : (memref<32x32xbf16, 2>)
              %asyncToken_31, %valOut_32 = air.region async [%arg26]  : (!air.async.token) {
                %18 = memref.alloc() : memref<32x32xbf16, 2>
                air.region_terminator %18 : memref<32x32xbf16, 2>
              } {id = 13 : i32} : (memref<32x32xbf16, 2>)
              %15 = affine.if #set0()[%arg18, %arg19] -> !air.async.token {
                %c0_36 = arith.constant 0 : index
                %18 = air.dma_memcpy_nd async [%asyncToken_29, %arg26] (%valOut_30[] [] [], %arg22[%c0_36, %arg25] [%c32, %c32] [%c64_21, %c1_19]) {broadcast_set = #set0} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %18 : !air.async.token
              } else {
                %c32_36 = arith.constant 32 : index
                %18 = air.dma_memcpy_nd async [%asyncToken_29, %arg26] (%valOut_30[] [] [], %arg22[%c32_36, %arg25] [%c32, %c32] [%c64_21, %c1_19]) {broadcast_set = #set1} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %18 : !air.async.token
              }
              %16 = affine.if #set2()[%arg18, %arg19] -> !air.async.token {
                %c0_36 = arith.constant 0 : index
                %18 = air.dma_memcpy_nd async [%asyncToken_31, %arg26] (%valOut_32[] [] [], %arg23[%arg25, %c0_36] [%c32, %c32] [%c64_21, %c1_19]) {broadcast_set = #set2} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %18 : !air.async.token
              } else {
                %c32_36 = arith.constant 32 : index
                %18 = air.dma_memcpy_nd async [%asyncToken_31, %arg26] (%valOut_32[] [] [], %arg23[%arg25, %c32_36] [%c32, %c32] [%c64_21, %c1_19]) {broadcast_set = #set3} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %18 : !air.async.token
              }
              %asyncToken_33 = air.region async [%16, %arg26, %15]  : (!air.async.token, !air.async.token, !air.async.token) {
                linalg.matmul ins(%valOut_30, %valOut_32 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%valOut_27 : memref<32x32xbf16, 2>)
                air.region_terminator
              } {id = 15 : i32}
              %asyncToken_34 = air.region async [%asyncToken_33]  : (!air.async.token) {
                memref.dealloc %valOut_30 : memref<32x32xbf16, 2>
                air.region_terminator
              } {id = 16 : i32}
              %asyncToken_35 = air.region async [%asyncToken_33]  : (!air.async.token) {
                memref.dealloc %valOut_32 : memref<32x32xbf16, 2>
                air.region_terminator
              } {id = 17 : i32}
              %17 = air.wait_all async [%asyncToken_33, %asyncToken_34, %asyncToken_35] 
              scf.yield %17 : !air.async.token
            }
            %14 = air.dma_memcpy_nd async [%13] (%arg24[%valOut_23, %valOut_25] [%c32, %c32] [%c64_21, %c1_19], %valOut_27[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            %asyncToken_28 = air.region async [%14]  : (!air.async.token) {
              memref.dealloc %valOut_27 : memref<32x32xbf16, 2>
              air.region_terminator
            } {id = 18 : i32}
            air.herd_terminator
          }
          %asyncToken_17 = air.region async [%9]  : (!air.async.token) {
            memref.dealloc %valOut_14 : memref<64x64xbf16, 1>
            air.region_terminator
          } {id = 19 : i32}
          %asyncToken_18 = air.region async [%9]  : (!air.async.token) {
            memref.dealloc %valOut_16 : memref<64x64xbf16, 1>
            air.region_terminator
          } {id = 20 : i32}
          %10 = air.wait_all async [%9, %asyncToken_17, %asyncToken_18] 
          scf.yield %10 : !air.async.token
        }
        %6 = air.dma_memcpy_nd async [%5] (%arg15[%valOut_7, %valOut_9] [%c64, %c64] [%c1024, %c1], %valOut_11[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
        %asyncToken_12 = air.region async [%6]  : (!air.async.token) {
          memref.dealloc %valOut_11 : memref<64x64xbf16, 1>
          air.region_terminator
        } {id = 21 : i32}
        air.partition_terminator
      }
      air.launch_terminator
    }
    %asyncToken_4, %valOut_5 = air.region async  {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.region_terminator %2 : memref<24576x1024xbf16>
    } {id = 22 : i32} : (memref<24576x1024xbf16>)
    %1 = air.launch async [%asyncToken_4, %0] (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%valOut_2, %arg7=%valOut_5) : memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {id = 6 : i32} {
      %2 = air.partition async  args(%arg8=%arg2, %arg9=%arg3, %arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7) : index, index, index, index, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {id = 5 : i32} {
        %c1 = arith.constant 1 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %c2 = arith.constant 2 : index
        %asyncToken_6, %valOut_7 = air.region async  {
          %6 = affine.apply #map0()[%arg8]
          air.region_terminator %6 : index
        } {id = 23 : i32} : (index)
        %asyncToken_8, %valOut_9 = air.region async  {
          %6 = affine.apply #map0()[%arg9]
          air.region_terminator %6 : index
        } {id = 24 : i32} : (index)
        %asyncToken_10, %valOut_11 = air.region async  {
          %6 = memref.alloc() : memref<64x64xbf16, 1>
          air.region_terminator %6 : memref<64x64xbf16, 1>
        } {id = 25 : i32} : (memref<64x64xbf16, 1>)
        %asyncToken_12, %valOut_13 = air.region async  {
          %6 = memref.alloc() : memref<64x64xbf16, 1>
          air.region_terminator %6 : memref<64x64xbf16, 1>
        } {id = 26 : i32} : (memref<64x64xbf16, 1>)
        %3 = air.dma_memcpy_nd async [%asyncToken_10, %asyncToken_8, %asyncToken_6] (%valOut_11[] [] [], %arg12[%valOut_7, %valOut_9] [%c64, %c64] [%c1024, %c1]) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        %4 = air.herd @herd_1 async [%3]  tile (%arg14, %arg15) in (%arg16=%c2, %arg17=%c2) args(%arg18=%valOut_11, %arg19=%valOut_13) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 4 : i32} {
          %c1_16 = arith.constant 1 : index
          %c64_17 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %cst_18 = arith.constant 2.000000e+00 : bf16
          %cst_19 = arith.constant 1.000000e+00 : bf16
          %cst_20 = arith.constant 5.000000e-01 : bf16
          %asyncToken_21, %valOut_22 = air.region async  {
            %8 = affine.apply #map1()[%arg14]
            air.region_terminator %8 : index
          } {id = 27 : i32} : (index)
          %asyncToken_23, %valOut_24 = air.region async  {
            %8 = affine.apply #map1()[%arg15]
            air.region_terminator %8 : index
          } {id = 28 : i32} : (index)
          %asyncToken_25, %valOut_26 = air.region async  {
            %8 = memref.alloc() : memref<32x32xbf16, 2>
            air.region_terminator %8 : memref<32x32xbf16, 2>
          } {id = 29 : i32} : (memref<32x32xbf16, 2>)
          %asyncToken_27, %valOut_28 = air.region async  {
            %8 = memref.alloc() : memref<32x32xbf16, 2>
            air.region_terminator %8 : memref<32x32xbf16, 2>
          } {id = 30 : i32} : (memref<32x32xbf16, 2>)
          %6 = air.dma_memcpy_nd async [%asyncToken_25, %asyncToken_23, %asyncToken_21] (%valOut_26[] [] [], %arg18[%valOut_22, %valOut_24] [%c32, %c32] [%c64_17, %c1_16]) {id = 11 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          %asyncToken_29 = air.region async [%6]  : (!air.async.token) {
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%valOut_26 : memref<32x32xbf16, 2>) outs(%valOut_28 : memref<32x32xbf16, 2>) {
            ^bb0(%arg20: bf16, %arg21: bf16):
              %8 = math.sqrt %cst_18 : bf16
              %9 = arith.divf %arg20, %8 : bf16
              %10 = math.erf %9 : bf16
              %11 = arith.addf %10, %cst_19 : bf16
              %12 = arith.mulf %11, %cst_20 : bf16
              %13 = arith.mulf %arg20, %12 : bf16
              linalg.yield %13 : bf16
            }
            air.region_terminator
          } {id = 31 : i32}
          %7 = air.dma_memcpy_nd async [%asyncToken_29] (%arg19[%valOut_22, %valOut_24] [%c32, %c32] [%c64_17, %c1_16], %valOut_28[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          %asyncToken_30 = air.region async [%asyncToken_29]  : (!air.async.token) {
            memref.dealloc %valOut_26 : memref<32x32xbf16, 2>
            air.region_terminator
          } {id = 32 : i32}
          %asyncToken_31 = air.region async [%7]  : (!air.async.token) {
            memref.dealloc %valOut_28 : memref<32x32xbf16, 2>
            air.region_terminator
          } {id = 33 : i32}
          air.herd_terminator
        }
        %5 = air.dma_memcpy_nd async [%4] (%arg13[%valOut_7, %valOut_9] [%c64, %c64] [%c1024, %c1], %valOut_13[] [] []) {id = 14 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
        %asyncToken_14 = air.region async [%4]  : (!air.async.token) {
          memref.dealloc %valOut_11 : memref<64x64xbf16, 1>
          air.region_terminator
        } {id = 34 : i32}
        %asyncToken_15 = air.region async [%5]  : (!air.async.token) {
          memref.dealloc %valOut_13 : memref<64x64xbf16, 1>
          air.region_terminator
        } {id = 35 : i32}
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %valOut_5 : memref<24576x1024xbf16>
  }
}
