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
    %async_token, %results = air.execute -> (memref<24576x1024xbf16>) {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.execute_terminator %2 : memref<24576x1024xbf16>
    }
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%cst : bf16) outs(%results : memref<24576x1024xbf16>)
    }
    %async_token_1, %results_2 = air.execute -> (memref<24576x1024xbf16>) {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.execute_terminator %2 : memref<24576x1024xbf16>
    }
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
    }
    %0 = air.launch async [%async_token_3] (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%results_2) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> attributes {id = 1 : i32} {
      %2 = air.partition async  args(%arg9=%arg2, %arg10=%arg3, %arg11=%arg4, %arg12=%arg5, %arg13=%arg6, %arg14=%arg7, %arg15=%arg8) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> attributes {id = 2 : i32} {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %async_token_6, %results_7 = air.execute -> (index) {
          %7 = affine.apply #map0()[%arg9]
          air.execute_terminator %7 : index
        }
        %async_token_8, %results_9 = air.execute -> (index) {
          %7 = affine.apply #map0()[%arg10]
          air.execute_terminator %7 : index
        }
        %3 = air.wait_all async [%async_token_8, %async_token_6] 
        %async_token_10, %results_11 = air.execute -> (memref<64x64xbf16, 1>) {
          %7 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %7 : memref<64x64xbf16, 1>
        }
        %4 = air.dma_memcpy_nd async [%async_token_10, %3] (%results_11[] [] [], %arg15[%results_7, %results_9] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        %5 = scf.for %arg16 = %c0 to %c1024 step %c64 iter_args(%arg17 = %4) -> (!air.async.token) {
          %async_token_13, %results_14 = air.execute [%arg17] -> (memref<64x64xbf16, 1>) {
            %11 = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %11 : memref<64x64xbf16, 1>
          }
          %async_token_15, %results_16 = air.execute [%arg17] -> (memref<64x64xbf16, 1>) {
            %11 = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %11 : memref<64x64xbf16, 1>
          }
          %7 = air.dma_memcpy_nd async [%async_token_13] (%results_14[] [] [], %arg13[%results_7, %arg16] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          %8 = air.dma_memcpy_nd async [%async_token_15] (%results_16[] [] [], %arg14[%arg16, %results_9] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          %9 = air.herd @herd_0 async [%8, %7]  tile (%arg18, %arg19) in (%arg20=%c2, %arg21=%c2) args(%arg22=%results_14, %arg23=%results_16, %arg24=%results_11) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 3 : i32} {
            %c1_19 = arith.constant 1 : index
            %c0_20 = arith.constant 0 : index
            %c64_21 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %async_token_22, %results_23 = air.execute -> (index) {
              %15 = affine.apply #map1()[%arg18]
              air.execute_terminator %15 : index
            }
            %async_token_24, %results_25 = air.execute -> (index) {
              %15 = affine.apply #map1()[%arg19]
              air.execute_terminator %15 : index
            }
            %11 = air.wait_all async [%async_token_24, %async_token_22] 
            %async_token_26, %results_27 = air.execute -> (memref<32x32xbf16, 2>) {
              %15 = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %15 : memref<32x32xbf16, 2>
            }
            %12 = air.dma_memcpy_nd async [%async_token_26, %11] (%results_27[] [] [], %arg24[%results_23, %results_25] [%c32, %c32] [%c64_21, %c1_19]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            %13 = scf.for %arg25 = %c0_20 to %c64_21 step %c32 iter_args(%arg26 = %12) -> (!air.async.token) {
              %async_token_29, %results_30 = air.execute [%arg26] -> (memref<32x32xbf16, 2>) {
                %18 = memref.alloc() : memref<32x32xbf16, 2>
                air.execute_terminator %18 : memref<32x32xbf16, 2>
              }
              %async_token_31, %results_32 = air.execute [%arg26] -> (memref<32x32xbf16, 2>) {
                %18 = memref.alloc() : memref<32x32xbf16, 2>
                air.execute_terminator %18 : memref<32x32xbf16, 2>
              }
              %15 = affine.if #set0()[%arg18, %arg19] -> !air.async.token {
                %c0_36 = arith.constant 0 : index
                %18 = air.dma_memcpy_nd async [%async_token_29] (%results_30[] [] [], %arg22[%c0_36, %arg25] [%c32, %c32] [%c64_21, %c1_19]) {broadcast_set = #set0, id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %18 : !air.async.token
              } else {
                %c32_36 = arith.constant 32 : index
                %18 = air.dma_memcpy_nd async [%async_token_29] (%results_30[] [] [], %arg22[%c32_36, %arg25] [%c32, %c32] [%c64_21, %c1_19]) {broadcast_set = #set1, id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %18 : !air.async.token
              }
              %16 = affine.if #set2()[%arg18, %arg19] -> !air.async.token {
                %c0_36 = arith.constant 0 : index
                %18 = air.dma_memcpy_nd async [%async_token_31] (%results_32[] [] [], %arg23[%arg25, %c0_36] [%c32, %c32] [%c64_21, %c1_19]) {broadcast_set = #set2, id = 7 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %18 : !air.async.token
              } else {
                %c32_36 = arith.constant 32 : index
                %18 = air.dma_memcpy_nd async [%async_token_31] (%results_32[] [] [], %arg23[%arg25, %c32_36] [%c32, %c32] [%c64_21, %c1_19]) {broadcast_set = #set3, id = 8 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                affine.yield %18 : !air.async.token
              }
              %async_token_33 = air.execute [%16, %15] {
                linalg.matmul ins(%results_30, %results_32 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%results_27 : memref<32x32xbf16, 2>)
              }
              %async_token_34 = air.execute [%async_token_33] {
                memref.dealloc %results_30 : memref<32x32xbf16, 2>
              }
              %async_token_35 = air.execute [%async_token_33] {
                memref.dealloc %results_32 : memref<32x32xbf16, 2>
              }
              %17 = air.wait_all async [%async_token_35, %async_token_34] 
              scf.yield %17 : !air.async.token
            }
            %14 = air.dma_memcpy_nd async [%13] (%arg24[%results_23, %results_25] [%c32, %c32] [%c64_21, %c1_19], %results_27[] [] []) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            %async_token_28 = air.execute [%14] {
              memref.dealloc %results_27 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          %async_token_17 = air.execute [%9] {
            memref.dealloc %results_14 : memref<64x64xbf16, 1>
          }
          %async_token_18 = air.execute [%9] {
            memref.dealloc %results_16 : memref<64x64xbf16, 1>
          }
          %10 = air.wait_all async [%async_token_18, %async_token_17] 
          scf.yield %10 : !air.async.token
        }
        %6 = air.dma_memcpy_nd async [%5] (%arg15[%results_7, %results_9] [%c64, %c64] [%c1024, %c1], %results_11[] [] []) {id = 10 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
        %async_token_12 = air.execute [%6] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1>
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    %async_token_4, %results_5 = air.execute -> (memref<24576x1024xbf16>) {
      %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
      air.execute_terminator %2 : memref<24576x1024xbf16>
    }
    %1 = air.launch async [%async_token_4, %0] (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%results_2, %arg7=%results_5) : memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {id = 4 : i32} {
      %2 = air.partition async  args(%arg8=%arg2, %arg9=%arg3, %arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7) : index, index, index, index, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {id = 5 : i32} {
        %c1 = arith.constant 1 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %c2 = arith.constant 2 : index
        %async_token_6, %results_7 = air.execute -> (index) {
          %6 = affine.apply #map0()[%arg8]
          air.execute_terminator %6 : index
        }
        %async_token_8, %results_9 = air.execute -> (index) {
          %6 = affine.apply #map0()[%arg9]
          air.execute_terminator %6 : index
        }
        %async_token_10, %results_11 = air.execute -> (memref<64x64xbf16, 1>) {
          %6 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %6 : memref<64x64xbf16, 1>
        }
        %async_token_12, %results_13 = air.execute -> (memref<64x64xbf16, 1>) {
          %6 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %6 : memref<64x64xbf16, 1>
        }
        %3 = air.dma_memcpy_nd async [%async_token_10, %async_token_8, %async_token_6] (%results_11[] [] [], %arg12[%results_7, %results_9] [%c64, %c64] [%c1024, %c1]) {id = 11 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        %4 = air.herd @herd_1 async [%async_token_12, %3]  tile (%arg14, %arg15) in (%arg16=%c2, %arg17=%c2) args(%arg18=%results_11, %arg19=%results_13) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 6 : i32} {
          %c1_16 = arith.constant 1 : index
          %c64_17 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %cst_18 = arith.constant 2.000000e+00 : bf16
          %cst_19 = arith.constant 1.000000e+00 : bf16
          %cst_20 = arith.constant 5.000000e-01 : bf16
          %async_token_21, %results_22 = air.execute -> (index) {
            %8 = affine.apply #map1()[%arg14]
            air.execute_terminator %8 : index
          }
          %async_token_23, %results_24 = air.execute -> (index) {
            %8 = affine.apply #map1()[%arg15]
            air.execute_terminator %8 : index
          }
          %async_token_25, %results_26 = air.execute -> (memref<32x32xbf16, 2>) {
            %8 = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %8 : memref<32x32xbf16, 2>
          }
          %async_token_27, %results_28 = air.execute -> (memref<32x32xbf16, 2>) {
            %8 = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %8 : memref<32x32xbf16, 2>
          }
          %6 = air.dma_memcpy_nd async [%async_token_25, %async_token_23, %async_token_21] (%results_26[] [] [], %arg18[%results_22, %results_24] [%c32, %c32] [%c64_17, %c1_16]) {id = 12 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          %async_token_29 = air.execute [%async_token_27, %6] {
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%results_26 : memref<32x32xbf16, 2>) outs(%results_28 : memref<32x32xbf16, 2>) {
            ^bb0(%arg20: bf16, %arg21: bf16):
              %8 = math.sqrt %cst_18 : bf16
              %9 = arith.divf %arg20, %8 : bf16
              %10 = math.erf %9 : bf16
              %11 = arith.addf %10, %cst_19 : bf16
              %12 = arith.mulf %11, %cst_20 : bf16
              %13 = arith.mulf %arg20, %12 : bf16
              linalg.yield %13 : bf16
            }
          }
          %7 = air.dma_memcpy_nd async [%async_token_29] (%arg19[%results_22, %results_24] [%c32, %c32] [%c64_17, %c1_16], %results_28[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          %async_token_30 = air.execute [%async_token_29] {
            memref.dealloc %results_26 : memref<32x32xbf16, 2>
          }
          %async_token_31 = air.execute [%7] {
            memref.dealloc %results_28 : memref<32x32xbf16, 2>
          }
          air.herd_terminator
        }
        %5 = air.dma_memcpy_nd async [%4] (%arg13[%results_7, %results_9] [%c64, %c64] [%c1024, %c1], %results_13[] [] []) {id = 14 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
        %async_token_14 = air.execute [%4] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1>
        }
        %async_token_15 = air.execute [%5] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1>
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %results_5 : memref<24576x1024xbf16>
  }
}
