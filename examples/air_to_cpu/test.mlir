
module attributes {torch.debug_module_name = "mmult"}  {
  func.func @forward(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<?x?xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<1024x1024xi32>
    affine.for %arg3 = 0 to 1024 {
      affine.for %arg4 = 0 to 1024 {
        affine.store %c0_i32, %0[%arg3, %arg4] : memref<1024x1024xi32>
      }
    }
    %1 = memref.cast %arg2 : memref<?x?xi32> to memref<1024x1024xi32>
    affine.for %arg3 = 0 to 1024 {
      affine.for %arg4 = 0 to 1024 {
        %2 = affine.load %0[%arg3, %arg4] : memref<1024x1024xi32>
        affine.store %2, %arg2[%arg3, %arg4] : memref<?x?xi32>
      }
    }
    scf.for %arg3 = %c0 to %c1024 step %c1024 {
      scf.for %arg4 = %c0 to %c1024 step %c512 {
        %init_event = air.wait_all async
        %e0, %e1 = scf.for %arg5 = %c0 to %c1024 step %c64 iter_args(%last_herd_event = %init_event, %last_dma_event = %init_event) -> (!air.async.token, !air.async.token) {
          %2 = memref.alloc() : memref<64x64xi32, 1>
          %3 = memref.alloc() : memref<64x64xi32, 1>
          %4 = memref.alloc() : memref<64x64xi32, 1>
          %dma_event0 = air.dma_memcpy_nd async [%last_herd_event] (%2[] [] [], %arg0[%arg3, %arg5] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xi32, 1>, memref<1024x1024xi32>)
          %dma_event1 = air.dma_memcpy_nd async [%last_herd_event] (%3[] [] [], %arg1[%arg5, %arg4] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xi32, 1>, memref<1024x1024xi32>)
          %dma_event2 = air.dma_memcpy_nd async [%last_herd_event] (%4[] [] [], %1[%arg3, %arg4] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<1024x1024xi32>)
          air.herd tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%2, %arg11=%3, %arg12=%4) : memref<64x64xi32, 1>,memref<64x64xi32, 1>,memref<64x64xi32, 1>attributes {sym_name = "herd_0"} {
            %c32 = arith.constant 32 : index
            %c0_0 = arith.constant 0 : index
            %c64_1 = arith.constant 64 : index
            %c1_2 = arith.constant 1 : index
            %5 = arith.muli %arg6, %c32 : index
            %6 = arith.muli %arg7, %c32 : index
            scf.for %arg13 = %c0_0 to %c64_1 step %c32 {
              %7 = memref.alloc() : memref<32x32xi32, 2>
              %8 = memref.alloc() : memref<32x32xi32, 2>
              %9 = memref.alloc() : memref<32x32xi32, 2>
              air.dma_memcpy_nd (%7[] [] [], %arg10[%5, %arg13] [%c32, %c32] [%c64_1, %c1_2]) {id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              air.dma_memcpy_nd (%8[] [] [], %arg11[%arg13, %6] [%c32, %c32] [%c64_1, %c1_2]) {id = 5 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              air.dma_memcpy_nd (%9[] [] [], %arg12[%5, %6] [%c32, %c32] [%c64_1, %c1_2]) {id = 6 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
              affine.for %arg14 = 0 to 32 {
                affine.for %arg15 = 0 to 32 {
                  affine.for %arg16 = 0 to 32 {
                    %10 = affine.load %7[%arg14, %arg16] : memref<32x32xi32, 2>
                    %11 = affine.load %8[%arg16, %arg15] : memref<32x32xi32, 2>
                    %12 = affine.load %9[%arg14, %arg15] : memref<32x32xi32, 2>
                    %13 = arith.muli %10, %11 : i32
                    %14 = arith.addi %12, %13 : i32
                    affine.store %14, %9[%arg14, %arg15] : memref<32x32xi32, 2>
                  }
                }
              }
              air.dma_memcpy_nd (%arg12[%5, %6] [%c32, %c32] [%c64_1, %c1_2], %9[] [] []) {id = 7 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
              memref.dealloc %7 : memref<32x32xi32, 2>
              memref.dealloc %8 : memref<32x32xi32, 2>
              memref.dealloc %9 : memref<32x32xi32, 2>
            }
            air.herd_terminator
          }
          %dma_event3 = air.dma_memcpy_nd async [%dma_event2] (%1[%arg3, %arg4] [%c64, %c64] [%c1024, %c1], %4[] [] []) {id = 8 : i32} : (memref<1024x1024xi32>, memref<64x64xi32, 1>)
          memref.dealloc %2 : memref<64x64xi32, 1>
          memref.dealloc %3 : memref<64x64xi32, 1>
          memref.dealloc %4 : memref<64x64xi32, 1>
          scf.yield %dma_event3, %dma_event3 : !air.async.token, !air.async.token
        }
        air.wait_all [%e0, %e1]
      }
    }
    return
  }
}

