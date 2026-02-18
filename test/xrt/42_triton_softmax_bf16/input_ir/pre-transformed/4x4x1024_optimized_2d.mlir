// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Optimized softmax IR with reductions hoisted outside loops
// 2D VERSION: 4×4 herd (16 cores) for improved parallelism
// Loop 1 & 2: Vector elementwise accumulation (NO reduction inside loop)
// Post-loop: Single vector.reduction operation

module {
  func.func @softmax_kernel(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c16_i32 = arith.constant 16 : i32
    %0 = arith.muli %arg5, %c16_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %1, %c1024 : index
    // 2D: Reinterpret as 4×4×1024 (16 rows arranged as 4×4 grid)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [4, 4, 1024], strides: [4096, 1024, 1] : memref<*xbf16> to memref<4x4x1024xbf16, strided<[4096, 1024, 1], offset: ?>>
    %alloc = memref.alloc() : memref<4x4x1024xbf16, 1 : i32>
    memref.copy %reinterpret_cast, %alloc : memref<4x4x1024xbf16, strided<[4096, 1024, 1], offset: ?>> to memref<4x4x1024xbf16, 1 : i32>
    %alloc_0 = memref.alloc() : memref<4x4x1024xbf16, 1>
    // 2D: 4×4 herd (16 cores total)
    air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) args(%arg12=%alloc, %arg13=%alloc_0) : memref<4x4x1024xbf16, 1 : i32>, memref<4x4x1024xbf16, 1> {
      %cst = arith.constant 1.000000e+00 : f32
      %cst_2 = arith.constant 0.000000e+00 : bf16
      %cst_neg_inf = arith.constant 0xFF80 : bf16  // -inf in bf16
      %cst_neg_inf_vec = arith.constant dense<0xFF80> : vector<32xbf16>
      %cst_zero_vec = arith.constant dense<0.000000e+00> : vector<32xbf16>
      %3 = ub.poison : bf16
      %c1_4 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024_5 = arith.constant 1024 : index
      %c32 = arith.constant 32 : index
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xbf16, 2>
      // 2D: DMA uses both %arg8 and %arg9 for 2D indexing
      air.dma_memcpy_nd (%alloc_6[] [] [], %arg12[%arg8, %arg9, %c0] [%c1_4, %c1_4, %c1024_5] [%c1024_5, %c1024_5, %c1_4]) {id = 1 : i32} : (memref<1x1024xbf16, 2>, memref<4x4x1024xbf16, 1 : i32>)
      
      // ============================================================
      // PHASE 1: Max reduction - OPTIMIZED with vector accumulation
      // ============================================================
      // Loop with iter_args: accumulate elementwise max as vector
      %max_vec_final = scf.for %arg14 = %c0 to %c1024_5 step %c32 iter_args(%acc_max_vec = %cst_neg_inf_vec) -> vector<32xbf16> {
        %subview = memref.subview %alloc_6[0, %arg14] [1, 32] [1, 1] : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
        %chunk = vector.transfer_read %subview[%c0, %c0], %3 {in_bounds = [true]} : memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>, vector<32xbf16>
        // Elementwise max - NO REDUCTION inside loop!
        %new_acc = arith.maximumf %chunk, %acc_max_vec : vector<32xbf16>
        scf.yield %new_acc : vector<32xbf16>
      }
      // POST-LOOP: Single reduction from vector to scalar
      %max_scalar = vector.reduction <maxnumf>, %max_vec_final : vector<32xbf16> into bf16
      %max_scalar_f32 = arith.extf %max_scalar : bf16 to f32
      
      // Store max for later use
      %alloc_7 = memref.alloc() : memref<1xf32, 2>
      memref.store %max_scalar_f32, %alloc_7[%c0] : memref<1xf32, 2>
      
      // ============================================================
      // PHASE 2: Sum of exp(x - max) - OPTIMIZED with vector accumulation
      // ============================================================
      // Loop with iter_args: accumulate exp values elementwise
      %sum_vec_final = scf.for %arg14 = %c0 to %c1024_5 step %c32 iter_args(%acc_sum_vec = %cst_zero_vec) -> vector<32xbf16> {
        %subview = memref.subview %alloc_6[0, %arg14] [1, 32] [1, 1] : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
        %chunk = vector.transfer_read %subview[%c0, %c0], %3 {in_bounds = [true]} : memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>, vector<32xbf16>
        // Compute exp(x - max)
        %chunk_f32 = arith.extf %chunk : vector<32xbf16> to vector<32xf32>
        %max_bcast = vector.broadcast %max_scalar_f32 : f32 to vector<32xf32>
        %diff = arith.subf %chunk_f32, %max_bcast : vector<32xf32>
        %diff_bf16 = arith.truncf %diff : vector<32xf32> to vector<32xbf16>
        %exp_val = math.exp %diff_bf16 : vector<32xbf16>
        // Elementwise add - NO REDUCTION inside loop!
        %new_acc = arith.addf %exp_val, %acc_sum_vec : vector<32xbf16>
        scf.yield %new_acc : vector<32xbf16>
      }
      // POST-LOOP: Single reduction from vector to scalar
      %sum_scalar = vector.reduction <add>, %sum_vec_final : vector<32xbf16> into bf16
      %sum_scalar_f32 = arith.extf %sum_scalar : bf16 to f32
      
      // Store sum for later use
      %alloc_8 = memref.alloc() : memref<1xf32, 2>
      memref.store %sum_scalar_f32, %alloc_8[%c0] : memref<1xf32, 2>
      
      // ============================================================
      // PHASE 3: Compute final output: exp(x - max) / sum
      // ============================================================
      %alloc_9 = memref.alloc() : memref<1x1024xbf16, 2>
      %inv_sum_f32 = arith.divf %cst, %sum_scalar_f32 : f32
      %inv_sum = arith.truncf %inv_sum_f32 : f32 to bf16
      %inv_sum_vec = vector.broadcast %inv_sum : bf16 to vector<32xbf16>
      
      scf.for %arg14 = %c0 to %c1024_5 step %c32 {
        %subview = memref.subview %alloc_6[0, %arg14] [1, 32] [1, 1] : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
        %subview_10 = memref.subview %alloc_9[0, %arg14] [1, 32] [1, 1] : memref<1x1024xbf16, 2> to memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
        %chunk = vector.transfer_read %subview[%c0, %c0], %3 {in_bounds = [true]} : memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>, vector<32xbf16>
        // Compute exp(x - max)
        %chunk_f32 = arith.extf %chunk : vector<32xbf16> to vector<32xf32>
        %max_bcast = vector.broadcast %max_scalar_f32 : f32 to vector<32xf32>
        %diff = arith.subf %chunk_f32, %max_bcast : vector<32xf32>
        %diff_bf16 = arith.truncf %diff : vector<32xf32> to vector<32xbf16>
        %exp_val = math.exp %diff_bf16 : vector<32xbf16>
        // Multiply by 1/sum
        %result = arith.mulf %exp_val, %inv_sum_vec : vector<32xbf16>
        vector.transfer_write %result, %subview_10[%c0, %c0] {in_bounds = [true]} : vector<32xbf16>, memref<1x32xbf16, strided<[1024, 1], offset: ?>, 2>
      }
      
      memref.dealloc %alloc_7 : memref<1xf32, 2>
      memref.dealloc %alloc_8 : memref<1xf32, 2>
      // 2D: DMA uses both %arg8 and %arg9 for 2D indexing
      air.dma_memcpy_nd (%arg13[%arg8, %arg9, %c0] [%c1_4, %c1_4, %c1024_5] [%c1024_5, %c1024_5, %c1_4], %alloc_9[] [] []) {id = 2 : i32} : (memref<4x4x1024xbf16, 1>, memref<1x1024xbf16, 2>)
      memref.dealloc %alloc_9 : memref<1x1024xbf16, 2>
    }
    // 2D: Reinterpret output as 4×4×1024
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [4, 4, 1024], strides: [4096, 1024, 1] : memref<*xbf16> to memref<4x4x1024xbf16, strided<[4096, 1024, 1], offset: ?>>
    memref.copy %alloc_0, %reinterpret_cast_1 : memref<4x4x1024xbf16, 1> to memref<4x4x1024xbf16, strided<[4096, 1024, 1], offset: ?>>
    memref.dealloc %alloc_0 : memref<4x4x1024xbf16, 1>
    return
  }
}
