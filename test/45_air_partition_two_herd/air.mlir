#map0 = affine_map<()[s0] -> (s0 * 5120)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {torch.debug_module_name = "test"} {
  func.func @forward(%arg0: memref<10240xi32>, %arg1: memref<10240xi32>, %arg2: memref<10240xi32>) -> memref<10240xi32> {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<10240xi32>
    air.partition @part0  args(%arg3=%arg0, %arg4=%arg1, %arg5=%arg2, %arg6=%0) : memref<10240xi32>, memref<10240xi32>, memref<10240xi32>, memref<10240xi32> {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %1 = memref.alloc() {alignment = 128 : i64} : memref<10240xi32>
      // %1 = %arg1 + % arg2
      air.herd  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c1) args(%arg11=%arg4, %arg12=%arg5, %arg13=%1) : memref<10240xi32>, memref<10240xi32>, memref<10240xi32> attributes {sym_name = "herd_0", x_loc = 4 : i64, y_loc = 2 : i64} {
        %c1_0 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c5120 = arith.constant 5120 : index
        %c1280 = arith.constant 1280 : index
        %2 = affine.apply #map0()[%arg7]
        scf.for %arg14 = %c0 to %c5120 step %c1280 {
          %3 = arith.addi %2, %arg14 : index
          %4 = memref.alloc() : memref<1280xi32, 2>
          %5 = memref.alloc() : memref<1280xi32, 2>
          %6 = memref.alloc() : memref<1280xi32, 2>
          air.dma_memcpy_nd (%4[] [] [], %arg11[%3] [%c1280] [%c1_0]) {id = 1 : i32} : (memref<1280xi32, 2>, memref<10240xi32>)
          air.dma_memcpy_nd (%5[] [] [], %arg12[%3] [%c1280] [%c1_0]) {id = 2 : i32} : (memref<1280xi32, 2>, memref<10240xi32>)
          // air.dma_memcpy_nd (%6[] [] [], %arg13[%3] [%c1280] [%c1_0]) {id = 3 : i32} : (memref<1280xi32, 2>, memref<10240xi32>)
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%4, %5 : memref<1280xi32, 2>, memref<1280xi32, 2>) outs(%6 : memref<1280xi32, 2>) {
          ^bb0(%arg15: i32, %arg16: i32, %arg17: i32):
            %7 = arith.addi %arg15, %arg16 : i32
            linalg.yield %7 : i32
          }
          air.dma_memcpy_nd (%arg13[%3] [%c1280] [%c1_0], %6[] [] []) {id = 4 : i32} : (memref<10240xi32>, memref<1280xi32, 2>)
          memref.dealloc %4 : memref<1280xi32, 2>
          memref.dealloc %5 : memref<1280xi32, 2>
          memref.dealloc %6 : memref<1280xi32, 2>
        }
        air.herd_terminator
      }
      // %0 = %arg2 * %1
      air.herd  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c1) args(%arg11=%arg3, %arg12=%1, %arg13=%arg6) : memref<10240xi32>, memref<10240xi32>, memref<10240xi32> attributes {sym_name = "herd_1", x_loc = 10 : i64, y_loc = 2 : i64} {
        %c1_0 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c5120 = arith.constant 5120 : index
        %c1280 = arith.constant 1280 : index
        %2 = affine.apply #map0()[%arg7]
        scf.for %arg14 = %c0 to %c5120 step %c1280 {
          %3 = arith.addi %2, %arg14 : index
          %4 = memref.alloc() : memref<1280xi32, 2>
          %5 = memref.alloc() : memref<1280xi32, 2>
          %6 = memref.alloc() : memref<1280xi32, 2>
          air.dma_memcpy_nd (%4[] [] [], %arg11[%3] [%c1280] [%c1_0]) {id = 5 : i32} : (memref<1280xi32, 2>, memref<10240xi32>)
          air.dma_memcpy_nd (%5[] [] [], %arg12[%3] [%c1280] [%c1_0]) {id = 6 : i32} : (memref<1280xi32, 2>, memref<10240xi32>)
          // air.dma_memcpy_nd (%6[] [] [], %arg13[%3] [%c1280] [%c1_0]) {id = 7 : i32} : (memref<1280xi32, 2>, memref<10240xi32>)
          linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%4, %5 : memref<1280xi32, 2>, memref<1280xi32, 2>) outs(%6 : memref<1280xi32, 2>) {
          ^bb0(%arg15: i32, %arg16: i32, %arg17: i32):
            %7 = arith.muli %arg15, %arg16 : i32
            linalg.yield %7 : i32
          }
          air.dma_memcpy_nd (%arg13[%3] [%c1280] [%c1_0], %6[] [] []) {id = 8 : i32} : (memref<10240xi32>, memref<1280xi32, 2>)
          memref.dealloc %4 : memref<1280xi32, 2>
          memref.dealloc %5 : memref<1280xi32, 2>
          memref.dealloc %6 : memref<1280xi32, 2>
        }
        air.herd_terminator
      }
      air.partition_terminator
    }
    return %0 : memref<10240xi32>
  }
}

