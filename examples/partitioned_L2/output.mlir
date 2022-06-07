#map = affine_map<()[s0] -> (s0 * 32)>
#set0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0)>
#set2 = affine_set<()[s0, s1] : (s1 == 0, s0 >= 0)>
#set3 = affine_set<()[s0, s1] : (s1 - 1 == 0, s0 >= 0)>
module {
  func @matmul(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() {alignment = 128 : i64} : memref<512x512xbf16>
    memref.copy %arg2, %0 : memref<512x512xbf16> to memref<512x512xbf16>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c64, %c64) {
      scf.for %arg5 = %c0 to %c512 step %c32 {
        %1 = memref.alloc() : memref<32x32xbf16, 1>
        air.dma_memcpy_nd (%1[] [] [], %arg0[%arg3, %arg5] [%c32, %c32] [%c512, %c1]) {id = 1 : i32} : (memref<32x32xbf16, 1>, memref<512x512xbf16>)
        %2 = memref.alloc() : memref<32x32xbf16, 1>
        %3 = arith.addi %arg3, %c32 : index
        air.dma_memcpy_nd (%2[] [] [], %arg0[%3, %arg5] [%c32, %c32] [%c512, %c1]) {id = 1 : i32} : (memref<32x32xbf16, 1>, memref<512x512xbf16>)
        %4 = memref.alloc() : memref<32x32xbf16, 1>
        air.dma_memcpy_nd (%4[] [] [], %arg1[%arg5, %arg4] [%c32, %c32] [%c512, %c1]) {id = 2 : i32} : (memref<32x32xbf16, 1>, memref<512x512xbf16>)
        %5 = memref.alloc() : memref<32x32xbf16, 1>
        %6 = arith.addi %arg4, %c32 : index
        air.dma_memcpy_nd (%5[] [] [], %arg1[%arg5, %6] [%c32, %c32] [%c512, %c1]) {id = 2 : i32} : (memref<32x32xbf16, 1>, memref<512x512xbf16>)
        air.launch_herd  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%arg3, %arg11=%arg4, %arg12=%0, %arg13=%1, %arg14=%2, %arg15=%4, %arg16=%5) : index, index, memref<512x512xbf16>, memref<32x32xbf16, 1>, memref<32x32xbf16, 1>, memref<32x32xbf16, 1>, memref<32x32xbf16, 1> {
          %c32_0 = arith.constant 32 : index
          %c512_1 = arith.constant 512 : index
          %c1_2 = arith.constant 1 : index
          %7 = affine.apply #map()[%arg6]
          %8 = affine.apply #map()[%arg7]
          %9 = arith.addi %arg10, %7 : index
          %10 = arith.addi %arg11, %8 : index
          %11 = memref.alloc() : memref<32x32xbf16, 2>
          %12 = memref.alloc() : memref<32x32xbf16, 2>
          %13 = memref.alloc() : memref<32x32xbf16, 2>
          affine.if #set0()[%arg6, %arg7] {
            air.dma_memcpy_nd (%11[] [] [], %arg13[] [] []) : (memref<32x32xbf16, 2>, memref<32x32xbf16, 1>)
          }
          affine.if #set1()[%arg6, %arg7] {
            air.dma_memcpy_nd (%11[] [] [], %arg14[] [] []) : (memref<32x32xbf16, 2>, memref<32x32xbf16, 1>)
          }
          affine.if #set2()[%arg6, %arg7] {
            air.dma_memcpy_nd (%12[] [] [], %arg15[] [] []) : (memref<32x32xbf16, 2>, memref<32x32xbf16, 1>)
          }
          affine.if #set3()[%arg6, %arg7] {
            air.dma_memcpy_nd (%12[] [] [], %arg16[] [] []) : (memref<32x32xbf16, 2>, memref<32x32xbf16, 1>)
          }
          air.dma_memcpy_nd (%13[] [] [], %arg12[%9, %10] [%c32_0, %c32_0] [%c512_1, %c1_2]) {id = 3 : i32} : (memref<32x32xbf16, 2>, memref<512x512xbf16>)
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%11, %12 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%13 : memref<32x32xbf16, 2>)
          air.dma_memcpy_nd (%arg12[%9, %10] [%c32_0, %c32_0] [%c512_1, %c1_2], %13[] [] []) {id = 4 : i32} : (memref<512x512xbf16>, memref<32x32xbf16, 2>)
          memref.dealloc %11 : memref<32x32xbf16, 2>
          memref.dealloc %12 : memref<32x32xbf16, 2>
          memref.dealloc %13 : memref<32x32xbf16, 2>
          air.herd_terminator
        }
      }
      scf.yield
    }
    return
  }
}

<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
<block argument> of type 'index' at index: 6
<block argument> of type 'memref<512x512xbf16>' at index: 7
<block argument> of type 'index' at index: 8
<block argument> of type 'memref<512x512xbf16>' at index: 9
<block argument> of type 'memref<512x512xbf16>' at index: 10
<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
<block argument> of type 'index' at index: 6
<block argument> of type 'memref<512x512xbf16>' at index: 7
<block argument> of type 'index' at index: 8
<block argument> of type 'memref<512x512xbf16>' at index: 9
<block argument> of type 'memref<512x512xbf16>' at index: 10
<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
<block argument> of type 'index' at index: 6
<block argument> of type 'memref<512x512xbf16>' at index: 7
<block argument> of type 'index' at index: 8
<block argument> of type 'memref<512x512xbf16>' at index: 9
<block argument> of type 'memref<512x512xbf16>' at index: 10
<block argument> of type 'memref<32x32xbf16, 1>' at index: 11
<block argument> of type 'memref<32x32xbf16, 1>' at index: 12
<block argument> of type 'memref<32x32xbf16, 1>' at index: 13
<block argument> of type 'memref<32x32xbf16, 1>' at index: 14
<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
<block argument> of type 'index' at index: 6
<block argument> of type 'index' at index: 7
<block argument> of type 'memref<512x512xbf16>' at index: 8
<block argument> of type 'memref<32x32xbf16, 1>' at index: 9
<block argument> of type 'memref<32x32xbf16, 1>' at index: 10
<block argument> of type 'memref<32x32xbf16, 1>' at index: 11
<block argument> of type 'memref<32x32xbf16, 1>' at index: 12
<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
<block argument> of type 'memref<512x512xbf16>' at index: 6
<block argument> of type 'memref<32x32xbf16, 1>' at index: 7
<block argument> of type 'memref<32x32xbf16, 1>' at index: 8
<block argument> of type 'memref<32x32xbf16, 1>' at index: 9
<block argument> of type 'memref<32x32xbf16, 1>' at index: 10
<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
<block argument> of type 'memref<512x512xbf16>' at index: 6
<block argument> of type 'memref<32x32xbf16, 1>' at index: 7
<block argument> of type 'memref<32x32xbf16, 1>' at index: 8
<block argument> of type 'memref<32x32xbf16, 1>' at index: 9
<block argument> of type 'memref<32x32xbf16, 1>' at index: 10
<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
<block argument> of type 'memref<512x512xbf16>' at index: 6
<block argument> of type 'memref<32x32xbf16, 1>' at index: 7
<block argument> of type 'memref<32x32xbf16, 1>' at index: 8
<block argument> of type 'memref<32x32xbf16, 1>' at index: 9
<block argument> of type 'memref<32x32xbf16, 1>' at index: 10
<block argument> of type 'index' at index: 4
<block argument> of type 'index' at index: 5
<block argument> of type 'memref<512x512xbf16>' at index: 6
<block argument> of type 'memref<32x32xbf16, 1>' at index: 7
<block argument> of type 'memref<32x32xbf16, 1>' at index: 8
<block argument> of type 'memref<32x32xbf16, 1>' at index: 9
<block argument> of type 'memref<32x32xbf16, 1>' at index: 10
