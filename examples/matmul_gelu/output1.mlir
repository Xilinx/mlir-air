#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "mmult"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c16 = arith.constant 16 : index
    %c384 = arith.constant 384 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    linalg.fill ins(%cst : bf16) outs(%alloc : memref<24576x1024xbf16>)
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    memref.copy %alloc, %alloc_0 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
    air.launch (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc_0) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> {
      air.partition  args(%arg9=%arg2, %arg10=%arg3, %arg11=%arg4, %arg12=%arg5, %arg13=%arg6, %arg14=%arg7, %arg15=%arg8) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %0 = affine.apply #map()[%arg9]
        %1 = affine.apply #map()[%arg10]
        scf.for %arg16 = %c0 to %c1024 step %c64 {
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
          %alloc_4 = memref.alloc() : memref<64x64xbf16, 1>
          air.dma_memcpy_nd (%alloc_2[] [] [], %arg13[%0, %arg16] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg14[%arg16, %1] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg15[%0, %1] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
          air.herd @herd_0  tile (%arg17, %arg18) in (%arg19=%c2, %arg20=%c2) args(%arg21=%alloc_2, %arg22=%alloc_3, %arg23=%alloc_4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_5 = arith.constant 1 : index
            %c0_6 = arith.constant 0 : index
            %c64_7 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
            %2 = affine.apply #map1()[%arg17]
            %3 = affine.apply #map1()[%arg18]
            scf.for %arg24 = %c0_6 to %c64_7 step %c32 {
              %alloc_8 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
              %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
              air.dma_memcpy_nd (%alloc_8[] [] [], %arg21[%2, %arg24] [%c32, %c32] [%c64_7, %c1_5]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_9[] [] [], %arg22[%arg24, %3] [%c32, %c32] [%c64_7, %c1_5]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              air.dma_memcpy_nd (%alloc_10[] [] [], %arg23[%2, %3] [%c32, %c32] [%c64_7, %c1_5]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              linalg.matmul ins(%alloc_8, %alloc_9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>)
              air.dma_memcpy_nd (%arg23[%2, %3] [%c32, %c32] [%c64_7, %c1_5], %alloc_10[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
              memref.dealloc %alloc_8 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
              memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%arg15[%0, %1] [%c64, %c64] [%c1024, %c1], %alloc_4[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
          memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
          memref.dealloc %alloc_4 : memref<64x64xbf16, 1>
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    air.launch (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%alloc_0, %arg7=%alloc_1) : memref<24576x1024xbf16>, memref<24576x1024xbf16> {
      air.partition  args(%arg8=%arg2, %arg9=%arg3, %arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7) : index, index, index, index, memref<24576x1024xbf16>, memref<24576x1024xbf16> {
        %c1 = arith.constant 1 : index
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %c2 = arith.constant 2 : index
        %0 = affine.apply #map()[%arg8]
        %1 = affine.apply #map()[%arg9]
        %alloc_2 = memref.alloc() : memref<64x64xbf16, 1>
        %alloc_3 = memref.alloc() : memref<64x64xbf16, 1>
        air.dma_memcpy_nd (%alloc_2[] [] [], %arg12[%0, %1] [%c64, %c64] [%c1024, %c1]) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        air.dma_memcpy_nd (%alloc_3[] [] [], %arg13[%0, %1] [%c64, %c64] [%c1024, %c1]) {id = 10 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        air.herd @herd_1  tile (%arg14, %arg15) in (%arg16=%c2, %arg17=%c2) args(%arg18=%alloc_2, %arg19=%alloc_3) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
          %c1_4 = arith.constant 1 : index
          %c64_5 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %cst_6 = arith.constant 2.000000e+00 : bf16
          %cst_7 = arith.constant 1.000000e+00 : bf16
          %cst_8 = arith.constant 5.000000e-01 : bf16
          %2 = affine.apply #map1()[%arg14]
          %3 = affine.apply #map1()[%arg15]
          %alloc_9 = memref.alloc() : memref<32x32xbf16, 2>
          %alloc_10 = memref.alloc() : memref<32x32xbf16, 2>
          air.dma_memcpy_nd (%alloc_9[] [] [], %arg18[%2, %3] [%c32, %c32] [%c64_5, %c1_4]) {id = 11 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          air.dma_memcpy_nd (%alloc_10[] [] [], %arg19[%2, %3] [%c32, %c32] [%c64_5, %c1_4]) {id = 12 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%alloc_9 : memref<32x32xbf16, 2>) outs(%alloc_10 : memref<32x32xbf16, 2>) {
          ^bb0(%in: bf16, %out: bf16):
            %4 = math.sqrt %cst_6 : bf16
            %5 = arith.divf %in, %4 : bf16
            %6 = math.erf %5 : bf16
            %7 = arith.addf %6, %cst_7 : bf16
            %8 = arith.mulf %7, %cst_8 : bf16
            %9 = arith.mulf %in, %8 : bf16
            linalg.yield %9 : bf16
          }
          air.dma_memcpy_nd (%arg19[%2, %3] [%c32, %c32] [%c64_5, %c1_4], %alloc_10[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          memref.dealloc %alloc_9 : memref<32x32xbf16, 2>
          memref.dealloc %alloc_10 : memref<32x32xbf16, 2>
          air.herd_terminator
        }
        air.dma_memcpy_nd (%arg13[%0, %1] [%c64, %c64] [%c1024, %c1], %alloc_3[] [] []) {id = 14 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
        memref.dealloc %alloc_2 : memref<64x64xbf16, 1>
        memref.dealloc %alloc_3 : memref<64x64xbf16, 1>
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %alloc_1 : memref<24576x1024xbf16>
  }
}
