// RUN: air-opt %s | FileCheck %s
module {

// CHECK-LABEL: module
// CHECK: func @foo
func @foo(%arg0 : memref<16x16xf32>, %arg1 : memref<16x16xf32>) -> () {
  %cst1 = constant 1 : index

  air.launch_herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) args(%ext0 = %arg0, %ext1 = %arg1) : memref<16x16xf32>, memref<16x16xf32> attributes { "foo" = "bar" } {
    %c0 = constant 0 : index
    %c256 = constant 256 : index
    %src0 = memref.alloc() : memref<16x16xf32, 2>
    %dst0 = memref.alloc() : memref<16x16xf32, 2>
    %e0 = air.wait_all async
    %e = air.dma_memcpy_2d async [%e0] (%ext0, %src0, [%c0, %c0], [%c0, %c0], %c256, %c256, %c256) : (memref<16x16xf32>, memref<16x16xf32, 2>, [index, index], [index, index], index, index, index) -> ()
    affine.for %arg4 = 0 to 16 {
      affine.for %arg5 = 0 to 16 {
        %0 = affine.load %src0[%arg4, %arg5] : memref<16x16xf32, 2>
        %cst = constant 1.000000e+00 : f32
        %1 = addf %0, %cst : f32
        affine.store %1, %dst0[%arg4, %arg5] : memref<16x16xf32, 2>
      }
    }
    "air.dma_memcpy_2d"(%dst0, %ext1, %c0, %c0, %c0, %c0, %c256, %c256, %c256) : (memref<16x16xf32, 2>, memref<16x16xf32>, index, index, index, index, index, index, index) -> ()
    air.herd_terminator
  }
  return
}

}
