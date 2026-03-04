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
      %c1_2 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : f32
      %thread_id_x = gpu.thread_id  x
      %0 = arith.remsi %thread_id_x, %c128 : index
      %1 = arith.divsi %thread_id_x, %c128 : index
      %2 = affine.apply #map()[%arg4]
      %3 = affine.apply #map()[%arg3]
      scf.for %arg20 = %c0 to %c64 step %c1_1 {
        memref.store %cst, %arg19[%arg20] : memref<64xf32, 5>
      }
      scf.for %arg20 = %c0 to %c4096 step %c8 {
        scf.for %arg21 = %c0 to %c4 step %c1_2 {
          %4 = arith.addi %2, %0 : index
          %5 = arith.muli %1, %c4 : index
          %6 = arith.addi %5, %arg21 : index
          %7 = arith.addi %6, %arg20 : index
          %8 = arith.remsi %4, %c128 : index
          %9 = arith.remsi %7, %c8 : index
          %10 = memref.load %arg0[%4, %7] : memref<4096x4096xf32>
          memref.store %10, %arg15[%8, %9] : memref<128x8xf32, 3>
        }
        scf.for %arg21 = %c0 to %c4 step %c1_2 {
          %4 = arith.addi %3, %0 : index
          %5 = arith.muli %1, %c4 : index
          %6 = arith.addi %5, %arg21 : index
          %7 = arith.addi %6, %arg20 : index
          %8 = arith.remsi %4, %c128 : index
          %9 = arith.remsi %7, %c8 : index
          %10 = memref.load %arg1[%7, %4] : memref<4096x4096xf32>
          memref.store %10, %arg16[%9, %8] : memref<8x128xf32, 3>
        }
        gpu.barrier
        %c0_3 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        %c16_5 = arith.constant 16 : index
        %c8_6 = arith.constant 8 : index
        scf.for %arg21 = %c0_3 to %c8_6 step %c1_4 {
          scf.for %arg22 = %c0_3 to %c8_6 step %c1_4 {
            %4 = arith.remsi %arg6, %c16_5 : index
            %5 = arith.muli %4, %c8_6 : index
            %6 = arith.addi %5, %arg22 : index
            %7 = memref.load %arg15[%6, %arg21] : memref<128x8xf32, 3>
            memref.store %7, %arg17[%arg22] : memref<8xf32, 5>
          }
          scf.for %arg22 = %c0_3 to %c8_6 step %c1_4 {
            %4 = arith.divsi %arg6, %c16_5 : index
            %5 = arith.muli %4, %c8_6 : index
            %6 = arith.addi %5, %arg22 : index
            %7 = memref.load %arg16[%arg21, %6] : memref<8x128xf32, 3>
            memref.store %7, %arg18[%arg22] : memref<8xf32, 5>
          }
          scf.for %arg22 = %c0_3 to %c8_6 step %c1_4 {
            scf.for %arg23 = %c0_3 to %c8_6 step %c1_4 {
              %4 = arith.muli %arg22, %c8_6 : index
              %5 = arith.addi %4, %arg23 : index
              %6 = memref.load %arg17[%arg22] : memref<8xf32, 5>
              %7 = memref.load %arg18[%arg23] : memref<8xf32, 5>
              %8 = memref.load %arg19[%5] : memref<64xf32, 5>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              memref.store %10, %arg19[%5] : memref<64xf32, 5>
            }
          }
        }
        gpu.barrier
        scf.for %arg21 = %c0 to %c8 step %c1_1 {
          scf.for %arg22 = %c0 to %c8 step %c1_1 {
            %4 = arith.muli %arg21, %c8 : index
            %5 = arith.addi %arg22, %4 : index
            %6 = memref.load %arg19[%5] : memref<64xf32, 5>
            %7 = arith.remsi %thread_id_x, %c16 : index
            %8 = arith.muli %7, %c8 : index
            %9 = arith.addi %2, %8 : index
            %10 = arith.addi %9, %arg21 : index
            %11 = arith.divsi %thread_id_x, %c16 : index
            %12 = arith.muli %11, %c8 : index
            %13 = arith.addi %3, %12 : index
            %14 = arith.addi %13, %arg22 : index
            memref.store %6, %arg2[%10, %14] : memref<4096x4096xf32>
          }
        }
      }
      gpu.terminator
    }
    return
  }
}

