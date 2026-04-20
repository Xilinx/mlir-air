#map = affine_map<()[s0] -> (s0 * 128)>
module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str0("Output match = %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("Val = %f:%f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("Input = %d:%d\0A\00") {addr_space = 0 : i32}
  llvm.func @mgpuStreamCreate() -> !llvm.ptr
  llvm.func @mgpuStreamDestroy(!llvm.ptr)
  llvm.func @mgpuEventSynchronize(!llvm.ptr)
  llvm.func @mgpuStreamSynchronize(!llvm.ptr)
  llvm.func @mgpuStreamWaitEvent(!llvm.ptr, !llvm.ptr)
  llvm.func @mgpuEventCreate() -> !llvm.ptr
  llvm.func @mgpuEventDestroy(!llvm.ptr)
  llvm.func @mgpuEventRecord(!llvm.ptr, !llvm.ptr)
  llvm.func @mgpuEventElapsedTime(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @mgpuCheckOutput(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
  llvm.func @mgpuInit(!llvm.ptr, !llvm.ptr, i64, i64)
  func.func @print_time(%arg0: f32) {
    %0 = llvm.mlir.constant(0 : i32) : i32
    return
  }
  func.func @main() {
    call @test_matmul() : () -> ()
    return
  }
  func.func @test_matmul() {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %alloc = memref.alloc() : memref<4096x4096xf32>
    %alloc_0 = memref.alloc() : memref<4096x4096xf32>
    %alloc_1 = memref.alloc() : memref<4096x4096xf32>
    %alloc_2 = memref.alloc() : memref<4096x4096xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<4096x4096xf32> -> index
    %intptr_4 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<4096x4096xf32> -> index
    %1 = arith.index_cast %intptr : index to i64
    %2 = arith.index_cast %intptr_4 : index to i64
    %3 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %4 = llvm.inttoptr %2 : i64 to !llvm.ptr
    %5 = arith.index_cast %c4096 : index to i64
    llvm.call @mgpuInit(%3, %4, %5, %5) : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()
    %memref = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref, %alloc : memref<4096x4096xf32>, memref<4096x4096xf32>
    %memref_5 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_5, %alloc_0 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %memref_6 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_6, %alloc_1 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %6 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %7 = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    %8 = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    llvm.call @mgpuEventRecord(%7, %6) : (!llvm.ptr, !llvm.ptr) -> ()
    call @forward(%memref, %memref_5, %memref_6) : (memref<4096x4096xf32>, memref<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    llvm.call @mgpuEventRecord(%8, %6) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @mgpuEventSynchronize(%8) : (!llvm.ptr) -> ()
    %c1_i32 = arith.constant 1 : i32
    %9 = llvm.alloca %c1_i32 x f32 : (i32) -> !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %10 = llvm.call @mgpuEventElapsedTime(%9, %7, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.call @mgpuStreamDestroy(%6) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%7) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%8) : (!llvm.ptr) -> ()
    gpu.memcpy  %alloc_1, %memref_6 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %intptr_7 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<4096x4096xf32> -> index
    %intptr_8 = memref.extract_aligned_pointer_as_index %alloc : memref<4096x4096xf32> -> index
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<4096x4096xf32> -> index
    %11 = arith.index_cast %intptr_7 : index to i64
    %12 = arith.index_cast %intptr_8 : index to i64
    %13 = arith.index_cast %intptr_9 : index to i64
    %14 = llvm.inttoptr %11 : i64 to !llvm.ptr
    %15 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %16 = llvm.inttoptr %13 : i64 to !llvm.ptr
    llvm.call @mgpuCheckOutput(%14, %15, %16, %5, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    return
  }
  func.func @forward(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c1_1 = arith.constant 1 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c32, %arg10 = %c32, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c256, %arg13 = %c1_1, %arg14 = %c1_0) workgroup(%arg15 : memref<128x8xf32, 3>, %arg16 : memref<8x128xf32, 3>) private(%arg17 : memref<8xf32, 5>, %arg18 : memref<8xf32, 5>, %arg19 : memref<64xf32, 5>) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map()[%arg3]
      scf.for %arg20 = %c0 to %c64 step %c1_1 {
        memref.store %cst, %arg19[%arg20] : memref<64xf32, 5>
      }
      scf.for %arg20 = %c0 to %c4096 step %c8 {
        %c0_7 = arith.constant 0 : index
        %c1_8 = arith.constant 1 : index
        %8 = arith.muli %c128, %c8 : index
        %thread_id_x_9 = gpu.thread_id x
        %block_dim_x_10 = gpu.block_dim x
        scf.for %arg21 = %thread_id_x_9 to %8 step %block_dim_x_10 {
          %c8_23 = arith.constant 8 : index
          %14 = arith.remsi %arg21, %c8_23 : index
          %15 = arith.divsi %arg21, %c8_23 : index
          %c128_24 = arith.constant 128 : index
          %16 = arith.remsi %15, %c128_24 : index
          %17 = arith.divsi %15, %c128_24 : index
          %18 = arith.addi %0, %16 : index
          %19 = arith.addi %arg20, %14 : index
          %20 = memref.load %arg0[%18, %19] : memref<4096x4096xf32>
          memref.store %20, %arg15[%16, %14] : memref<128x8xf32, 3>
        }
        gpu.barrier
        %c0_11 = arith.constant 0 : index
        %c1_12 = arith.constant 1 : index
        %9 = arith.muli %c8, %c128 : index
        %thread_id_x_13 = gpu.thread_id x
        %block_dim_x_14 = gpu.block_dim x
        scf.for %arg21 = %thread_id_x_13 to %9 step %block_dim_x_14 {
          %c128_23 = arith.constant 128 : index
          %14 = arith.remsi %arg21, %c128_23 : index
          %15 = arith.divsi %arg21, %c128_23 : index
          %c8_24 = arith.constant 8 : index
          %16 = arith.remsi %15, %c8_24 : index
          %17 = arith.divsi %15, %c8_24 : index
          %18 = arith.addi %arg20, %16 : index
          %19 = arith.addi %1, %14 : index
          %20 = memref.load %arg1[%18, %19] : memref<4096x4096xf32>
          memref.store %20, %arg16[%16, %14] : memref<8x128xf32, 3>
        }
        gpu.barrier
        gpu.barrier
        %thread_id_x_15 = gpu.thread_id x
        %thread_id_y_16 = gpu.thread_id y
        %block_dim_x_17 = gpu.block_dim x
        %block_dim_y_18 = gpu.block_dim y
        %c0_19 = arith.constant 0 : index
        %c1_20 = arith.constant 1 : index
        %c8_21 = arith.constant 8 : index
        %c16_22 = arith.constant 16 : index
        %10 = arith.remsi %thread_id_x_15, %c16_22 : index
        %11 = arith.divsi %thread_id_x_15, %c16_22 : index
        %12 = arith.muli %10, %c8_21 : index
        %13 = arith.muli %11, %c8_21 : index
        scf.for %arg21 = %c0_19 to %c8_21 step %c1_20 {
          %c0_23 = arith.constant 0 : index
          %c1_24 = arith.constant 1 : index
          scf.for %arg22 = %c0_23 to %c8_21 step %c1_24 {
            %c0_27 = arith.constant 0 : index
            %c8_28 = arith.constant 8 : index
            %14 = arith.muli %12, %c8_28 : index
            %15 = arith.addi %c0_27, %14 : index
            %c1_29 = arith.constant 1 : index
            %16 = arith.muli %arg21, %c1_29 : index
            %17 = arith.addi %15, %16 : index
            %c0_30 = arith.constant 0 : index
            %18 = arith.muli %arg22, %c8_21 : index
            %19 = arith.addi %c0_30, %18 : index
            %20 = arith.addi %17, %19 : index
            %c8_31 = arith.constant 8 : index
            %21 = arith.remsi %20, %c8_31 : index
            %22 = arith.divsi %20, %c8_31 : index
            %c128_32 = arith.constant 128 : index
            %23 = arith.remsi %22, %c128_32 : index
            %24 = arith.divsi %22, %c128_32 : index
            %25 = memref.load %arg15[%23, %21] : memref<128x8xf32, 3>
            memref.store %25, %arg17[%arg22] : memref<8xf32, 5>
          }
          %c0_25 = arith.constant 0 : index
          %c1_26 = arith.constant 1 : index
          scf.for %arg22 = %c0_25 to %c8_21 step %c1_26 {
            %c0_27 = arith.constant 0 : index
            %c128_28 = arith.constant 128 : index
            %14 = arith.muli %arg21, %c128_28 : index
            %15 = arith.addi %c0_27, %14 : index
            %c1_29 = arith.constant 1 : index
            %16 = arith.muli %13, %c1_29 : index
            %17 = arith.addi %15, %16 : index
            %c0_30 = arith.constant 0 : index
            %18 = arith.muli %arg22, %c1_20 : index
            %19 = arith.addi %c0_30, %18 : index
            %20 = arith.addi %17, %19 : index
            %c128_31 = arith.constant 128 : index
            %21 = arith.remsi %20, %c128_31 : index
            %22 = arith.divsi %20, %c128_31 : index
            %c8_32 = arith.constant 8 : index
            %23 = arith.remsi %22, %c8_32 : index
            %24 = arith.divsi %22, %c8_32 : index
            %25 = memref.load %arg16[%23, %21] : memref<8x128xf32, 3>
            memref.store %25, %arg18[%arg22] : memref<8xf32, 5>
          }
          scf.for %arg22 = %c0_19 to %c8_21 step %c1_20 {
            scf.for %arg23 = %c0_19 to %c8_21 step %c1_20 {
              %14 = arith.muli %arg22, %c8_21 : index
              %15 = arith.addi %14, %arg23 : index
              %16 = memref.load %arg17[%arg22] : memref<8xf32, 5>
              %17 = memref.load %arg18[%arg23] : memref<8xf32, 5>
              %18 = memref.load %arg19[%15] : memref<64xf32, 5>
              %19 = arith.mulf %16, %17 : f32
              %20 = arith.addf %18, %19 : f32
              memref.store %20, %arg19[%15] : memref<64xf32, 5>
            }
          }
        }
        gpu.barrier
      }
      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %block_dim_x = gpu.block_dim x
      %block_dim_y = gpu.block_dim y
      %c8_2 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c4096_3 = arith.constant 4096 : index
      %c1_4 = arith.constant 1 : index
      %2 = arith.remsi %thread_id_x, %c16 : index
      %3 = arith.divsi %thread_id_x, %c16 : index
      %4 = arith.muli %2, %c8_2 : index
      %5 = arith.muli %3, %c8_2 : index
      %6 = arith.addi %0, %4 : index
      %7 = arith.addi %1, %5 : index
      %c0_5 = arith.constant 0 : index
      %c1_6 = arith.constant 1 : index
      scf.for %arg20 = %c0_5 to %c8_2 step %c1_6 {
        scf.for %arg21 = %c0_5 to %c8_2 step %c1_6 {
          %c0_7 = arith.constant 0 : index
          %c8_8 = arith.constant 8 : index
          %8 = arith.muli %arg20, %c8_8 : index
          %9 = arith.addi %c0_7, %8 : index
          %c1_9 = arith.constant 1 : index
          %10 = arith.muli %arg21, %c1_9 : index
          %11 = arith.addi %9, %10 : index
          %c64_10 = arith.constant 64 : index
          %12 = arith.remsi %11, %c64_10 : index
          %13 = arith.divsi %11, %c64_10 : index
          %14 = arith.addi %6, %arg20 : index
          %15 = arith.addi %7, %arg21 : index
          %16 = memref.load %arg19[%12] : memref<64xf32, 5>
          memref.store %16, %arg2[%14, %15] : memref<4096x4096xf32>
        }
      }
      gpu.terminator
    }
    return
  }
}

