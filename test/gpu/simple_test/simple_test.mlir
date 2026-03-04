// Simple AIR test for GPU compilation - direct global memory access
module {
  func.func @simple_add(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) {
    %c1 = arith.constant 1 : index

    // Use air.launch to generate gpu.launch
    air.launch (%bx, %by) in (%nbx=%c1, %nby=%c1) args(%in0=%arg0, %in1=%arg1, %out=%arg2) : memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32> {
      // Compute: element-wise add directly on global memory
      affine.for %i = 0 to 16 {
        affine.for %j = 0 to 16 {
          %a = affine.load %in0[%i, %j] : memref<16x16xf32>
          %b = affine.load %in1[%i, %j] : memref<16x16xf32>
          %c = arith.addf %a, %b : f32
          affine.store %c, %out[%i, %j] : memref<16x16xf32>
        }
      }
      air.launch_terminator
    }
    return
  }
}
