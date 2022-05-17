#map0 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 256 + s0 + d1)>
module attributes {torch.debug_module_name = "mmult"} {
  func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c12 = arith.constant 12 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c256 = arith.constant 256 : index
    %c192 = arith.constant 192 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    %1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    linalg.fill ins(%cst : bf16) outs(%0 : memref<24576x1024xbf16>)
    memref.copy %0, %1 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
    scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
      %2 = arith.remsi %arg2, %c4 : index
      %3 = arith.divsi %arg2, %c4 : index
      %4 = arith.muli %2, %c256 : index
      %5 = arith.muli %3, %c192 : index
      %6 = memref.subview %arg0[%5, 0] [192, 1024] [1, 1] : memref<24576x1024xbf16> to memref<192x1024xbf16, #map0>
      %7 = memref.subview %arg1[0, %4] [1024, 256] [1, 1] : memref<1024x1024xbf16> to memref<1024x256xbf16, #map0>
      %8 = memref.subview %1[%5, %4] [192, 256] [1, 1] : memref<24576x1024xbf16> to memref<192x256xbf16, #map0>
      scf.for %arg3 = %c0 to %c1024 step %c256 {
        %9 = memref.alloc() : memref<192x256xbf16, 1>
        %10 = memref.alloc() : memref<256x256xbf16, 1>
        air.dma_memcpy_nd (%9[] [] [], %6[%c0, %arg3] [%c192, %c256] [%c1024, %c1]) {id = 1 : i32} : (memref<192x256xbf16, 1>, memref<192x1024xbf16, #map0>)
        air.dma_memcpy_nd (%10[] [] [], %7[%arg3, %c0] [%c256, %c256] [%c1024, %c1]) {id = 2 : i32} : (memref<256x256xbf16, 1>, memref<1024x256xbf16, #map0>)
        air.launch_herd  tile (%arg4, %arg5) in (%arg6=%c4, %arg7=%c12) args(%arg8=%9, %arg9=%10, %arg10=%8) : memref<192x256xbf16, 1>, memref<256x256xbf16, 1>, memref<192x256xbf16, #map0> attributes {sym_name = "herd_0"} {
          %c4_0 = arith.constant 4 : index
          %c64 = arith.constant 64 : index
          %11 = arith.remsi %arg6, %c4_0 : index
          %12 = arith.divsi %arg6, %c4_0 : index
          %13 = arith.muli %11, %c64 : index
          %14 = arith.muli %12, %c64 : index
          %15 = memref.subview %arg8[%14, 0] [64, 256] [1, 1] : memref<192x256xbf16, 1> to memref<64x256xbf16, #map1, 1>
          %16 = memref.subview %arg9[0, %13] [256, 64] [1, 1] : memref<256x256xbf16, 1> to memref<256x64xbf16, #map1, 1>
          %17 = memref.subview %arg10[%14, %13] [64, 64] [1, 1] : memref<192x256xbf16, #map0> to memref<64x64xbf16, #map0>
          %18 = memref.subview %15[0, %arg4] [64, 64] [1, 1] : memref<64x256xbf16, #map1, 1> to memref<64x64xbf16, #map1, 1>
          %19 = memref.subview %16[%arg4, 0] [64, 64] [1, 1] : memref<256x64xbf16, #map1, 1> to memref<64x64xbf16, #map1, 1>
          air.pipeline {direction = "horiz"} {
            %20 = air.pipeline.stage args(%arg11=%17) : memref<64x64xbf16, #map0> {
              linalg.matmul ins(%18, %19 : memref<64x64xbf16, #map1, 1>, memref<64x64xbf16, #map1, 1>) outs(%arg11 : memref<64x64xbf16, #map0>)
              air.pipeline.yield %arg11 : memref<64x64xbf16, #map0>
            } : memref<64x64xbf16, #map0>
            %21 = air.pipeline.stage args(%arg11=%20) : memref<64x64xbf16, #map0> {
              linalg.matmul ins(%18, %19 : memref<64x64xbf16, #map1, 1>, memref<64x64xbf16, #map1, 1>) outs(%arg11 : memref<64x64xbf16, #map0>)
              air.pipeline.yield %arg11 : memref<64x64xbf16, #map0>
            } : memref<64x64xbf16, #map0>
            %22 = air.pipeline.stage args(%arg11=%21) : memref<64x64xbf16, #map0> {
              linalg.matmul ins(%18, %19 : memref<64x64xbf16, #map1, 1>, memref<64x64xbf16, #map1, 1>) outs(%arg11 : memref<64x64xbf16, #map0>)
              air.pipeline.yield %arg11 : memref<64x64xbf16, #map0>
            } : memref<64x64xbf16, #map0>
            %23 = air.pipeline.stage args(%arg11=%22) : memref<64x64xbf16, #map0> {
              linalg.matmul ins(%18, %19 : memref<64x64xbf16, #map1, 1>, memref<64x64xbf16, #map1, 1>) outs(%arg11 : memref<64x64xbf16, #map0>)
              air.pipeline.yield %arg11 : memref<64x64xbf16, #map0>
            } : memref<64x64xbf16, #map0>
            air.pipeline.terminator
          }
          air.herd_terminator
        }
        memref.dealloc %9 : memref<192x256xbf16, 1>
        memref.dealloc %10 : memref<256x256xbf16, 1>
      }
      scf.yield
    }
    return %1 : memref<24576x1024xbf16>
  }
}

