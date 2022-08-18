#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#set0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c24576 = arith.constant 24576 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %asyncToken, %valOut = air.region async  {
      %3 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.region_terminator %3 : memref<24576x1024xbf16>
    } {id = 1 : i32} : (memref<24576x1024xbf16>)
    %asyncToken_0 = air.region async [%asyncToken]  : (!air.async.token) {
      linalg.fill ins(%cst : bf16) outs(%valOut : memref<24576x1024xbf16>)
      air.region_terminator
    } {id = 2 : i32}
    %asyncToken_1, %valOut_2 = air.region async  {
      %3 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.region_terminator %3 : memref<24576x1024xbf16>
    } {id = 3 : i32} : (memref<24576x1024xbf16>)
    %asyncToken_3 = air.region async [%asyncToken_1, %asyncToken_0]  : (!air.async.token, !air.async.token) {
      memref.copy %valOut, %valOut_2 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
      air.region_terminator
    } {id = 4 : i32}
    %0 = scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c24576, %c1024) step (%c64, %c64) init (%asyncToken_3) -> !air.async.token {
      %asyncToken_6, %valOut_7 = air.region async  {
        %6 = memref.alloc() : memref<64x64xbf16, 1>
        air.region_terminator %6 : memref<64x64xbf16, 1>
      } {id = 7 : i32} : (memref<64x64xbf16, 1>)
      %3 = air.dma_memcpy_nd async [%asyncToken_3, %asyncToken_6] (%valOut_7[] [] [], %valOut_2[%arg2, %arg3] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
      %4 = scf.for %arg4 = %c0 to %c1024 step %c64 iter_args(%arg5 = %3) -> (!air.async.token) {
        %asyncToken_9, %valOut_10 = air.region async [%arg5]  : (!air.async.token) {
          %10 = memref.alloc() : memref<64x64xbf16, 1>
          air.region_terminator %10 : memref<64x64xbf16, 1>
        } {id = 5 : i32} : (memref<64x64xbf16, 1>)
        %asyncToken_11, %valOut_12 = air.region async [%arg5]  : (!air.async.token) {
          %10 = memref.alloc() : memref<64x64xbf16, 1>
          air.region_terminator %10 : memref<64x64xbf16, 1>
        } {id = 6 : i32} : (memref<64x64xbf16, 1>)
        %6 = air.dma_memcpy_nd async [%asyncToken_9] (%valOut_10[] [] [], %arg0[%arg2, %arg4] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        %7 = air.dma_memcpy_nd async [%asyncToken_11] (%valOut_12[] [] [], %arg1[%arg4, %arg3] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
        %8 = air.launch_herd async [%7, %6, %arg5]  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%valOut_10, %arg11=%valOut_12, %arg12=%valOut_7) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 1 : i32, sym_name = "herd_0"} {
          %c1_15 = arith.constant 1 : index
          %c0_16 = arith.constant 0 : index
          %c64_17 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %asyncToken_18, %valOut_19 = air.region async  {
            %14 = affine.apply #map0()[%arg6]
            air.region_terminator %14 : index
          } {id = 8 : i32} : (index)
          %asyncToken_20, %valOut_21 = air.region async  {
            %14 = affine.apply #map0()[%arg7]
            air.region_terminator %14 : index
          } {id = 9 : i32} : (index)
          %10 = air.wait_all async [%asyncToken_18, %asyncToken_20] 
          %asyncToken_22, %valOut_23 = air.region async  {
            %14 = memref.alloc() : memref<32x32xbf16, 2>
            air.region_terminator %14 : memref<32x32xbf16, 2>
          } {id = 12 : i32} : (memref<32x32xbf16, 2>)
          %11 = air.dma_memcpy_nd async [%10, %asyncToken_22] (%valOut_23[] [] [], %arg12[%valOut_19, %valOut_21] [%c32, %c32] [%c64_17, %c1_15]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          %12 = scf.for %arg13 = %c0_16 to %c64_17 step %c32 iter_args(%arg14 = %11) -> (!air.async.token) {
            %asyncToken_25, %valOut_26 = air.region async [%arg14]  : (!air.async.token) {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              air.region_terminator %17 : memref<32x32xbf16, 2>
            } {id = 10 : i32} : (memref<32x32xbf16, 2>)
            %asyncToken_27, %valOut_28 = air.region async [%arg14]  : (!air.async.token) {
              %17 = memref.alloc() : memref<32x32xbf16, 2>
              air.region_terminator %17 : memref<32x32xbf16, 2>
            } {id = 11 : i32} : (memref<32x32xbf16, 2>)
            %14 = affine.if #set0()[%arg6, %arg7] -> !air.async.token {
              %c0_32 = arith.constant 0 : index
              %17 = air.dma_memcpy_nd async [%asyncToken_25, %arg14] (%valOut_26[] [] [], %arg10[%c0_32, %arg13] [%c32, %c32] [%c64_17, %c1_15]) {broadcast_set = #set0} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              affine.yield %17 : !air.async.token
            } else {
              %c32_32 = arith.constant 32 : index
              %17 = air.dma_memcpy_nd async [%asyncToken_25, %arg14] (%valOut_26[] [] [], %arg10[%c32_32, %arg13] [%c32, %c32] [%c64_17, %c1_15]) {broadcast_set = #set1} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              affine.yield %17 : !air.async.token
            }
            %15 = affine.if #set2()[%arg6, %arg7] -> !air.async.token {
              %c0_32 = arith.constant 0 : index
              %17 = air.dma_memcpy_nd async [%asyncToken_27, %arg14] (%valOut_28[] [] [], %arg11[%arg13, %c0_32] [%c32, %c32] [%c64_17, %c1_15]) {broadcast_set = #set2} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              affine.yield %17 : !air.async.token
            } else {
              %c32_32 = arith.constant 32 : index
              %17 = air.dma_memcpy_nd async [%asyncToken_27, %arg14] (%valOut_28[] [] [], %arg11[%arg13, %c32_32] [%c32, %c32] [%c64_17, %c1_15]) {broadcast_set = #set3} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              affine.yield %17 : !air.async.token
            }
            %asyncToken_29 = air.region async [%15, %arg14, %14]  : (!air.async.token, !air.async.token, !air.async.token) {
              linalg.matmul ins(%valOut_26, %valOut_28 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%valOut_23 : memref<32x32xbf16, 2>)
              air.region_terminator
            } {id = 13 : i32}
            %asyncToken_30 = air.region async [%asyncToken_29]  : (!air.async.token) {
              memref.dealloc %valOut_26 : memref<32x32xbf16, 2>
              air.region_terminator
            } {id = 14 : i32}
            %asyncToken_31 = air.region async [%asyncToken_29]  : (!air.async.token) {
              memref.dealloc %valOut_28 : memref<32x32xbf16, 2>
              air.region_terminator
            } {id = 15 : i32}
            %16 = air.wait_all async [%asyncToken_29, %asyncToken_30, %asyncToken_31] 
            scf.yield %16 : !air.async.token
          }
          %13 = air.dma_memcpy_nd async [%12] (%arg12[%valOut_19, %valOut_21] [%c32, %c32] [%c64_17, %c1_15], %valOut_23[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          %asyncToken_24 = air.region async [%13]  : (!air.async.token) {
            memref.dealloc %valOut_23 : memref<32x32xbf16, 2>
            air.region_terminator
          } {id = 16 : i32}
          air.herd_terminator
        }
        %asyncToken_13 = air.region async [%8]  : (!air.async.token) {
          memref.dealloc %valOut_10 : memref<64x64xbf16, 1>
          air.region_terminator
        } {id = 17 : i32}
        %asyncToken_14 = air.region async [%8]  : (!air.async.token) {
          memref.dealloc %valOut_12 : memref<64x64xbf16, 1>
          air.region_terminator
        } {id = 18 : i32}
        %9 = air.wait_all async [%8, %asyncToken_13, %asyncToken_14] 
        scf.yield %9 : !air.async.token
      }
      %5 = air.dma_memcpy_nd async [%4] (%valOut_2[%arg2, %arg3] [%c64, %c64] [%c1024, %c1], %valOut_7[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
      %asyncToken_8 = air.region async [%5]  : (!air.async.token) {
        memref.dealloc %valOut_7 : memref<64x64xbf16, 1>
        air.region_terminator
      } {id = 19 : i32}
      scf.reduce(%asyncToken_8)  : !air.async.token {
      ^bb0(%arg4: !air.async.token, %arg5: !air.async.token):
        %6 = air.wait_all async [%arg4, %arg5] 
        scf.reduce.return %6 : !air.async.token
      }
      scf.yield
    }
    %asyncToken_4, %valOut_5 = air.region async  {
      %3 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.region_terminator %3 : memref<24576x1024xbf16>
    } {id = 20 : i32} : (memref<24576x1024xbf16>)
    %1 = air.wait_all async [%0, %asyncToken_4] 
    %2 = scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c24576, %c1024) step (%c64, %c64) init (%1) -> !air.async.token {
      %asyncToken_6, %valOut_7 = air.region async [%1]  : (!air.async.token) {
        %7 = memref.alloc() : memref<64x64xbf16, 1>
        air.region_terminator %7 : memref<64x64xbf16, 1>
      } {id = 21 : i32} : (memref<64x64xbf16, 1>)
      %asyncToken_8, %valOut_9 = air.region async [%1]  : (!air.async.token) {
        %7 = memref.alloc() : memref<64x64xbf16, 1>
        air.region_terminator %7 : memref<64x64xbf16, 1>
      } {id = 22 : i32} : (memref<64x64xbf16, 1>)
      %3 = air.dma_memcpy_nd async [%asyncToken_6, %1] (%valOut_7[] [] [], %valOut_2[%arg2, %arg3] [%c64, %c64] [%c1024, %c1]) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
      %4 = air.launch_herd async [%3]  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%valOut_7, %arg9=%valOut_9) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 2 : i32, sym_name = "herd_1"} {
        %c1_12 = arith.constant 1 : index
        %c64_13 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %cst_14 = arith.constant 2.000000e+00 : bf16
        %cst_15 = arith.constant 1.000000e+00 : bf16
        %cst_16 = arith.constant 5.000000e-01 : bf16
        %asyncToken_17, %valOut_18 = air.region async  {
          %9 = affine.apply #map0()[%arg4]
          air.region_terminator %9 : index
        } {id = 23 : i32} : (index)
        %asyncToken_19, %valOut_20 = air.region async  {
          %9 = affine.apply #map0()[%arg5]
          air.region_terminator %9 : index
        } {id = 24 : i32} : (index)
        %asyncToken_21, %valOut_22 = air.region async  {
          %9 = memref.alloc() : memref<32x32xbf16, 2>
          air.region_terminator %9 : memref<32x32xbf16, 2>
        } {id = 25 : i32} : (memref<32x32xbf16, 2>)
        %asyncToken_23, %valOut_24 = air.region async  {
          %9 = memref.alloc() : memref<32x32xbf16, 2>
          air.region_terminator %9 : memref<32x32xbf16, 2>
        } {id = 26 : i32} : (memref<32x32xbf16, 2>)
        %7 = air.dma_memcpy_nd async [%asyncToken_21, %asyncToken_19, %asyncToken_17] (%valOut_22[] [] [], %arg8[%valOut_18, %valOut_20] [%c32, %c32] [%c64_13, %c1_12]) {id = 11 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
        %asyncToken_25 = air.region async [%7]  : (!air.async.token) {
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%valOut_22 : memref<32x32xbf16, 2>) outs(%valOut_24 : memref<32x32xbf16, 2>) {
          ^bb0(%arg10: bf16, %arg11: bf16):
            %9 = math.sqrt %cst_14 : bf16
            %10 = arith.divf %arg10, %9 : bf16
            %11 = math.erf %10 : bf16
            %12 = arith.addf %11, %cst_15 : bf16
            %13 = arith.mulf %12, %cst_16 : bf16
            %14 = arith.mulf %arg10, %13 : bf16
            linalg.yield %14 : bf16
          }
          air.region_terminator
        } {id = 27 : i32}
        %8 = air.dma_memcpy_nd async [%asyncToken_25] (%arg9[%valOut_18, %valOut_20] [%c32, %c32] [%c64_13, %c1_12], %valOut_24[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
        %asyncToken_26 = air.region async [%asyncToken_25]  : (!air.async.token) {
          memref.dealloc %valOut_22 : memref<32x32xbf16, 2>
          air.region_terminator
        } {id = 28 : i32}
        %asyncToken_27 = air.region async [%8]  : (!air.async.token) {
          memref.dealloc %valOut_24 : memref<32x32xbf16, 2>
          air.region_terminator
        } {id = 29 : i32}
        air.herd_terminator
      }
      %5 = air.dma_memcpy_nd async [%4] (%valOut_5[%arg2, %arg3] [%c64, %c64] [%c1024, %c1], %valOut_9[] [] []) {id = 14 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
      %asyncToken_10 = air.region async [%4]  : (!air.async.token) {
        memref.dealloc %valOut_7 : memref<64x64xbf16, 1>
        air.region_terminator
      } {id = 30 : i32}
      %asyncToken_11 = air.region async [%5]  : (!air.async.token) {
        memref.dealloc %valOut_9 : memref<64x64xbf16, 1>
        air.region_terminator
      } {id = 31 : i32}
      %6 = air.wait_all async [%asyncToken_10, %asyncToken_11] 
      scf.reduce(%6)  : !air.async.token {
      ^bb0(%arg4: !air.async.token, %arg5: !air.async.token):
        %7 = air.wait_all async [%arg4, %arg5] 
        scf.reduce.return %7 : !air.async.token
      }
      scf.yield
    }
    return %valOut_5 : memref<24576x1024xbf16>
  }
}
