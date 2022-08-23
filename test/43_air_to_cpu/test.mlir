// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

module attributes {torch.debug_module_name = "mmult"}  {
  func.func @forward(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<?x?xi32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() : memref<1024x1024xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<1024x1024xi32>)
    %1 = memref.cast %arg2 : memref<?x?xi32> to memref<1024x1024xi32>
    linalg.copy ins(%0 : memref<1024x1024xi32>) outs(%1 : memref<1024x1024xi32>)
    scf.for %arg3 = %c0 to %c1024 step %c64 {
      scf.for %arg4 = %c0 to %c1024 step %c64 {
        scf.for %arg5 = %c0 to %c1024 step %c64 {
          %2 = memref.alloc() : memref<64x64xi32, 1>
          %3 = memref.alloc() : memref<64x64xi32, 1>
          %4 = memref.alloc() : memref<64x64xi32, 1>
          air.dma_memcpy_nd (%2[] [] [], %arg0[%arg3, %arg5] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xi32, 1>, memref<1024x1024xi32>)
          air.dma_memcpy_nd (%3[] [] [], %arg1[%arg5, %arg4] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xi32, 1>, memref<1024x1024xi32>)
          air.dma_memcpy_nd (%4[] [] [], %1[%arg3, %arg4] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<1024x1024xi32>)
          air.herd tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%2, %arg11=%3, %arg12=%4) : memref<64x64xi32, 1>, memref<64x64xi32, 1>, memref<64x64xi32, 1> attributes {sym_name = "herd_0"} {
            %c1_0 = arith.constant 1 : index
            %c64_1 = arith.constant 64 : index
            %c0_2 = arith.constant 0 : index
            %c32 = arith.constant 32 : index
            %5 = arith.muli %arg6, %c32 : index
            %6 = arith.muli %arg7, %c32 : index
            scf.for %arg13 = %c0_2 to %c64_1 step %c32 {
              %7 = memref.alloc() : memref<32x32xi32, 2>
              %8 = memref.alloc() : memref<32x32xi32, 2>
              %9 = memref.alloc() : memref<32x32xi32, 2>
              air.dma_memcpy_nd (%7[] [] [], %arg10[%5, %arg13] [%c32, %c32] [%c64_1, %c1_0]) {id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              air.dma_memcpy_nd (%8[] [] [], %arg11[%arg13, %6] [%c32, %c32] [%c64_1, %c1_0]) {id = 5 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              air.dma_memcpy_nd (%9[] [] [], %arg12[%5, %6] [%c32, %c32] [%c64_1, %c1_0]) {id = 6 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              linalg.matmul ins(%7, %8 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%9 : memref<32x32xi32, 2>)
              air.dma_memcpy_nd (%arg12[%5, %6] [%c32, %c32] [%c64_1, %c1_0], %9[] [] []) {id = 7 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
              memref.dealloc %7 : memref<32x32xi32, 2>
              memref.dealloc %8 : memref<32x32xi32, 2>
              memref.dealloc %9 : memref<32x32xi32, 2>
            }
            air.herd_terminator
          }
          air.dma_memcpy_nd (%1[%arg3, %arg4] [%c64, %c64] [%c1024, %c1], %4[] [] []) {id = 8 : i32} : (memref<1024x1024xi32>, memref<64x64xi32, 1>)
          memref.dealloc %2 : memref<64x64xi32, 1>
          memref.dealloc %3 : memref<64x64xi32, 1>
          memref.dealloc %4 : memref<64x64xi32, 1>
        }
      }
    }
    return
  }
}
