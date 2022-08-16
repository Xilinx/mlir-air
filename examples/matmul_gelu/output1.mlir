#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c24576 = arith.constant 24576 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    linalg.fill ins(%cst : bf16) outs(%0 : memref<24576x1024xbf16>)
    %1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    memref.copy %0, %1 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c24576, %c1024) step (%c64, %c64) {
      scf.for %arg4 = %c0 to %c1024 step %c64 {
        %3 = memref.alloc() : memref<64x64xbf16, 1>
        %4 = memref.alloc() : memref<64x64xbf16, 1>
        %5 = memref.alloc() : memref<64x64xbf16, 1>
        air.dma_memcpy_nd (%3[] [] [], %arg0[%arg2, %arg4] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        air.dma_memcpy_nd (%4[] [] [], %arg1[%arg4, %arg3] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
        air.dma_memcpy_nd (%5[] [] [], %1[%arg2, %arg3] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        air.launch_herd  tile (%arg5, %arg6) in (%arg7=%c2, %arg8=%c2) args(%arg9=%3, %arg10=%4, %arg11=%5) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {sym_name = "herd_0"} {
          %c1_0 = arith.constant 1 : index
          %c0_1 = arith.constant 0 : index
          %c64_2 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %6 = affine.apply #map0()[%arg5]
          %7 = affine.apply #map0()[%arg6]
          scf.for %arg12 = %c0_1 to %c64_2 step %c32 {
            %8 = memref.alloc() : memref<32x32xbf16, 2>
            %9 = memref.alloc() : memref<32x32xbf16, 2>
            %10 = memref.alloc() : memref<32x32xbf16, 2>
            air.dma_memcpy_nd (%8[] [] [], %arg9[%6, %arg12] [%c32, %c32] [%c64_2, %c1_0]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%9[] [] [], %arg10[%arg12, %7] [%c32, %c32] [%c64_2, %c1_0]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            air.dma_memcpy_nd (%10[] [] [], %arg11[%6, %7] [%c32, %c32] [%c64_2, %c1_0]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
            linalg.matmul ins(%8, %9 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%10 : memref<32x32xbf16, 2>)
            air.dma_memcpy_nd (%arg11[%6, %7] [%c32, %c32] [%c64_2, %c1_0], %10[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
            memref.dealloc %8 : memref<32x32xbf16, 2>
            memref.dealloc %9 : memref<32x32xbf16, 2>
            memref.dealloc %10 : memref<32x32xbf16, 2>
          }
          air.herd_terminator
        }
        air.dma_memcpy_nd (%1[%arg2, %arg3] [%c64, %c64] [%c1024, %c1], %5[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
        memref.dealloc %3 : memref<64x64xbf16, 1>
        memref.dealloc %4 : memref<64x64xbf16, 1>
        memref.dealloc %5 : memref<64x64xbf16, 1>
      }
      scf.yield
    }
    %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    scf.for %arg2 = %c0 to %c24576 step %c64 {
      scf.for %arg3 = %c0 to %c1024 step %c64 {
        %3 = memref.alloc() : memref<64x64xbf16, 1>
        %4 = memref.alloc() : memref<64x64xbf16, 1>
        air.dma_memcpy_nd (%3[] [] [], %1[%arg2, %arg3] [%c64, %c64] [%c1024, %c1]) {id = 9 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        air.dma_memcpy_nd (%4[] [] [], %2[%arg2, %arg3] [%c64, %c64] [%c1024, %c1]) {id = 10 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
        air.launch_herd  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%3, %arg9=%4) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {sym_name = "herd_1"} {
          %c1_0 = arith.constant 1 : index
          %c64_1 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %cst_2 = arith.constant 2.000000e+00 : bf16
          %cst_3 = arith.constant 1.000000e+00 : bf16
          %cst_4 = arith.constant 5.000000e-01 : bf16
          %5 = affine.apply #map0()[%arg4]
          %6 = affine.apply #map0()[%arg5]
          %7 = memref.alloc() : memref<32x32xbf16, 2>
          %8 = memref.alloc() : memref<32x32xbf16, 2>
          air.dma_memcpy_nd (%7[] [] [], %arg8[%5, %6] [%c32, %c32] [%c64_1, %c1_0]) {id = 11 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          air.dma_memcpy_nd (%8[] [] [], %arg9[%5, %6] [%c32, %c32] [%c64_1, %c1_0]) {id = 12 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7 : memref<32x32xbf16, 2>) outs(%8 : memref<32x32xbf16, 2>) {
          ^bb0(%arg10: bf16, %arg11: bf16):
            %9 = math.sqrt %cst_2 : bf16
            %10 = arith.divf %arg10, %9 : bf16
            %11 = math.erf %10 : bf16
            %12 = arith.addf %11, %cst_3 : bf16
            %13 = arith.mulf %12, %cst_4 : bf16
            %14 = arith.mulf %arg10, %13 : bf16
            linalg.yield %14 : bf16
          }
          air.dma_memcpy_nd (%arg9[%5, %6] [%c32, %c32] [%c64_1, %c1_0], %8[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          memref.dealloc %7 : memref<32x32xbf16, 2>
          memref.dealloc %8 : memref<32x32xbf16, 2>
          air.herd_terminator
        }
        air.dma_memcpy_nd (%2[%arg2, %arg3] [%c64, %c64] [%c1024, %c1], %4[] [] []) {id = 14 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
        memref.dealloc %3 : memref<64x64xbf16, 1>
        memref.dealloc %4 : memref<64x64xbf16, 1>
      }
    }
    return %2 : memref<24576x1024xbf16>
  }
}
