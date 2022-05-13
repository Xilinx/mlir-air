#map0 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 256 + s0 + d1)>
module attributes {torch.debug_module_name = "mmult"} {
  func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c12 = arith.constant 12 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c192 = arith.constant 192 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
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
        %9 = memref.subview %6[0, %arg3] [192, 256] [1, 1] : memref<192x1024xbf16, #map0> to memref<192x256xbf16, #map0>
        %10 = memref.subview %7[%arg3, 0] [256, 256] [1, 1] : memref<1024x256xbf16, #map0> to memref<256x256xbf16, #map0>
        %11 = memref.alloc() : memref<192x256xbf16, 1>
        %12 = memref.alloc() : memref<256x256xbf16, 1>
        memref.copy %9, %11 : memref<192x256xbf16, #map0> to memref<192x256xbf16, 1>
        memref.copy %10, %12 : memref<256x256xbf16, #map0> to memref<256x256xbf16, 1>
        scf.parallel (%arg4) = (%c0) to (%c12) step (%c1) {
          %13 = arith.remsi %arg4, %c4 : index
          %14 = arith.divsi %arg4, %c4 : index
          %15 = arith.muli %13, %c64 : index
          %16 = arith.muli %14, %c64 : index
          %17 = memref.subview %11[%16, 0] [64, 256] [1, 1] : memref<192x256xbf16, 1> to memref<64x256xbf16, #map1, 1>
          %18 = memref.subview %12[0, %15] [256, 64] [1, 1] : memref<256x256xbf16, 1> to memref<256x64xbf16, #map1, 1>
          %19 = memref.subview %8[%16, %15] [64, 64] [1, 1] : memref<192x256xbf16, #map0> to memref<64x64xbf16, #map0>
          air.launch_herd  tile (%arg5, %arg6) in (%arg7=%c4, %arg8=%c1) args(%arg9=%17, %arg10=%18, %arg11=%19) : memref<64x256xbf16, #map1, 1>, memref<256x64xbf16, #map1, 1>, memref<64x64xbf16, #map0> {
            %20 = memref.subview %arg9[0, %arg5] [64, 64] [1, 1] : memref<64x256xbf16, #map1, 1> to memref<64x64xbf16, #map1, 1>
            %21 = memref.subview %arg10[%arg5, 0] [64, 64] [1, 1] : memref<256x64xbf16, #map1, 1> to memref<64x64xbf16, #map1, 1>
            air.pipeline {
              %22 = air.pipeline.stage args(%arg12=%arg11) : memref<64x64xbf16, #map0> {
                linalg.matmul ins(%20, %21 : memref<64x64xbf16, #map1, 1>, memref<64x64xbf16, #map1, 1>) outs(%arg12 : memref<64x64xbf16, #map0>)
                air.pipeline.yield %arg12 : memref<64x64xbf16, #map0>
              } : memref<64x64xbf16, #map0>
              %23 = air.pipeline.stage args(%arg12=%22) : memref<64x64xbf16, #map0> {
                linalg.matmul ins(%20, %21 : memref<64x64xbf16, #map1, 1>, memref<64x64xbf16, #map1, 1>) outs(%arg12 : memref<64x64xbf16, #map0>)
                air.pipeline.yield %arg12 : memref<64x64xbf16, #map0>
              } : memref<64x64xbf16, #map0>
              %24 = air.pipeline.stage args(%arg12=%23) : memref<64x64xbf16, #map0> {
                linalg.matmul ins(%20, %21 : memref<64x64xbf16, #map1, 1>, memref<64x64xbf16, #map1, 1>) outs(%arg12 : memref<64x64xbf16, #map0>)
                air.pipeline.yield %arg12 : memref<64x64xbf16, #map0>
              } : memref<64x64xbf16, #map0>
              %25 = air.pipeline.stage args(%arg12=%24) : memref<64x64xbf16, #map0> {
                linalg.matmul ins(%20, %21 : memref<64x64xbf16, #map1, 1>, memref<64x64xbf16, #map1, 1>) outs(%arg12 : memref<64x64xbf16, #map0>)
                air.pipeline.yield %arg12 : memref<64x64xbf16, #map0>
              } : memref<64x64xbf16, #map0>
              air.pipeline.terminator
            }
            air.herd_terminator
          }
          scf.yield
        }
        memref.dealloc %11 : memref<192x256xbf16, 1>
        memref.dealloc %12 : memref<256x256xbf16, 1>
      }
      scf.yield
    }
    return %1 : memref<24576x1024xbf16>
  }
}

