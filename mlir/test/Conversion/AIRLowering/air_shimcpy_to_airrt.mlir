// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-to-std | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 128)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0 * 16)>
module  {
  // CHECK: func.func @task2
  // CHECK: airrt.dma_memcpy_nd(%c1_i32, %{{.*}}, %{{.*}}, %arg0[%c0_i64, %{{.*}}, %{{.*}}, %{{.*}}], [%c1_i64, %{{.*}}, %{{.*}}, %{{.*}}], [%c0_i64, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<4096x1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
  // CHECK: airrt.dma_memcpy_nd(%c2_i32, %{{.*}}, %{{.*}}, %arg1[%c0_i64_2, %{{.*}}, %{{.*}}, %{{.*}}], [%c1_i64_3, %{{.*}}, %{{.*}}, %{{.*}}], [%c0_i64_2, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<4096x1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
  // CHECK: airrt.dma_memcpy_nd(%c3_i32, %{{.*}}, %{{.*}}, %arg2[%c0_i64_4, %{{.*}}, %{{.*}}, %{{.*}}], [%c1_i64_5, %{{.*}}, %{{.*}}, %{{.*}}], [%c0_i64_4, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<4096x1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
  // CHECK: airrt.dma_memcpy_nd(%c4_i32, %{{.*}}, %{{.*}}, %arg2[%c0_i64_6, %{{.*}}, %{{.*}}, %{{.*}}], [%c1_i64_7, %{{.*}}, %{{.*}}, %{{.*}}], [%c0_i64_6, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<4096x1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
  func.func @task2(%arg0: memref<4096x1024x512xi32>, %arg1: memref<4096x1024x512xi32>, %arg2: memref<4096x1024x512xi32>) {
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c4096 step %c128 {
      scf.for %arg4 = %c0 to %c1024 step %c128 {
        scf.for %arg5 = %c0 to %c512 step %c128 {
          air.herd tile (%arg6, %arg7) in (%arg8=%c4, %arg9=%c4) args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg0, %arg13=%arg5, %arg14=%arg1, %arg15=%arg2) : index,index,memref<4096x1024x512xi32>,index,memref<4096x1024x512xi32>,memref<4096x1024x512xi32>attributes {sym_name = "herd_0"} {
            %c1 = arith.constant 1 : index
            %c512_0 = arith.constant 512 : index
            %c524288 = arith.constant 524288 : index
            %c128_1 = arith.constant 128 : index
            %c32 = arith.constant 32 : index
            %0 = arith.muli %arg6, %c32 : index
            %1 = arith.muli %arg7, %c32 : index
            %2 = arith.addi %arg10, %0 : index
            %3 = arith.addi %arg11, %1 : index
            %4 = memref.alloc() : memref<32x32x128xi32, 2>
            %5 = memref.alloc() : memref<32x32x128xi32, 2>
            %6 = memref.alloc() : memref<32x32x128xi32, 2>
            air.dma_memcpy_nd (%4[] [] [], %arg12[%2, %3, %arg13] [%c32, %c32, %c128_1] [%c524288, %c512_0, %c1]) {id = 1 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
            air.dma_memcpy_nd (%5[] [] [], %arg14[%2, %3, %arg13] [%c32, %c32, %c128_1] [%c524288, %c512_0, %c1]) {id = 2 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
            air.dma_memcpy_nd (%6[] [] [], %arg15[%2, %3, %arg13] [%c32, %c32, %c128_1] [%c524288, %c512_0, %c1]) {id = 3 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
            air.dma_memcpy_nd (%arg15[%2, %3, %arg13] [%c32, %c32, %c128_1] [%c524288, %c512_0, %c1], %6[] [] []) {id = 4 : i32} : (memref<4096x1024x512xi32>, memref<32x32x128xi32, 2>)
            memref.dealloc %4 : memref<32x32x128xi32, 2>
            memref.dealloc %5 : memref<32x32x128xi32, 2>
            memref.dealloc %6 : memref<32x32x128xi32, 2>
            air.herd_terminator
          }
        }
      }
    }
    return
  }
}
