#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 4)>
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
    %2 = arith.index_cast %intptr : index to i64
    %3 = arith.index_cast %intptr_4 : index to i64
    %4 = llvm.inttoptr %2 : i64 to !llvm.ptr
    %5 = llvm.inttoptr %3 : i64 to !llvm.ptr
    %6 = arith.index_cast %c4096 : index to i64
    llvm.call @mgpuInit(%4, %5, %6, %6) : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()
    %memref = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref, %alloc : memref<4096x4096xf32>, memref<4096x4096xf32>
    %memref_5 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_5, %alloc_0 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %memref_6 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_6, %alloc_1 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %7 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %8 = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    %9 = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    llvm.call @mgpuEventRecord(%8, %7) : (!llvm.ptr, !llvm.ptr) -> ()
    call @forward(%memref, %memref_5, %memref_6) : (memref<4096x4096xf32>, memref<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    llvm.call @mgpuEventRecord(%9, %7) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @mgpuEventSynchronize(%9) : (!llvm.ptr) -> ()
    %c1_i32 = arith.constant 1 : i32
    %10 = llvm.alloca %c1_i32 x f32 : (i32) -> !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %11 = llvm.call @mgpuEventElapsedTime(%10, %8, %9) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.call @mgpuStreamDestroy(%7) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%8) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%9) : (!llvm.ptr) -> ()
    gpu.memcpy  %alloc_1, %memref_6 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %intptr_7 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<4096x4096xf32> -> index
    %intptr_8 = memref.extract_aligned_pointer_as_index %alloc : memref<4096x4096xf32> -> index
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<4096x4096xf32> -> index
    %12 = arith.index_cast %intptr_7 : index to i64
    %13 = arith.index_cast %intptr_8 : index to i64
    %14 = arith.index_cast %intptr_9 : index to i64
    %15 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %16 = llvm.inttoptr %13 : i64 to !llvm.ptr
    %17 = llvm.inttoptr %14 : i64 to !llvm.ptr
    llvm.call @mgpuCheckOutput(%15, %16, %17, %6, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    return
  }
  func.func @forward(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %c32 = arith.constant 32 : index
    air.launch (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<4096x4096xf32>, memref<4096x4096xf32>, memref<4096x4096xf32> {
      air.segment @forward_0  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<4096x4096xf32>, memref<4096x4096xf32>, memref<4096x4096xf32> {
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c128 = arith.constant 128 : index
        %c32_0 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %c4096 = arith.constant 4096 : index
        %c256 = arith.constant 256 : index
        %c8 = arith.constant 8 : index
        %c16_0 = arith.constant 16 : index
        %cst = arith.constant 0.000000e+00 : f32
        %tidx = gpu.thread_id  x
        %rBIdx = arith.remsi %tidx, %c128 : index
        %rBIdy = arith.divsi %tidx, %c128 : index
        %0 = affine.apply #map()[%arg11]
        %1 = affine.apply #map()[%arg10]
        %c64 = arith.constant 64 : index
        %arg118 = memref.alloc() : memref<8xf32, 2>
        %arg119 = memref.alloc() : memref<8xf32, 2>
        %arg120 = memref.alloc() : memref<64xf32, 2>
        scf.for %i = %c0 to %c64 step %c1 {
           memref.store %cst, %arg120[%i] : memref<64xf32, 2>
        }
        scf.for %arg15 = %c0 to %c4096 step %c8 {
          %alloc = memref.alloc() : memref<128x128xf32, 1>
          %alloc_1 = memref.alloc() : memref<128x8xf32, 1>
          %alloc_2 = memref.alloc() : memref<8x128xf32, 1>
          %c0_2 = arith.constant 0 : index
          %c1_3 = arith.constant 1 : index
          %c31 = arith.constant 31 : index
          //index_x = blk_x*128 + tid % 128
          //index_y = (tid / 128) * 4 + arg15 + arg23
          scf.for %arg23 = %c0 to %c4 step %c1_3 {
              %index_x = arith.addi %0, %rBIdx : index
              %2 = arith.muli %rBIdy, %c4 : index
              %4 = arith.addi %2, %arg23 : index
              %index_y = arith.addi %4, %arg15 : index
              %idx = arith.remsi %index_x, %c128 : index
              %idy = arith.remsi %index_y, %c8 : index
              %6 = memref.load %arg12[%index_x, %index_y] : memref<4096x4096xf32>
              memref.store %6, %alloc_1[%idx, %idy] : memref<128x8xf32, 1>
          }
          //index_y = (tid / 128) * 4 + arg15 + arg23
          //index_x = blk_y*128 + tid % 128
          scf.for %arg23 = %c0_2 to %c4 step %c1_3 {
              %index_x = arith.addi %1, %rBIdx : index
              %2 = arith.muli %rBIdy, %c4 : index
              %4 = arith.addi %2, %arg23 : index
              %index_y = arith.addi %4, %arg15 : index
              %idx = arith.remsi %index_x, %c128 : index
              %idy = arith.remsi %index_y, %c8 : index
              %6 = memref.load %arg13[%index_y, %index_x] : memref<4096x4096xf32>
              memref.store %6, %alloc_2[%idy, %idx] : memref<8x128xf32, 1>
          }
          gpu.barrier
          air.herd @herd_0  tile (%arg31, %arg32) in (%arg33=%c256, %arg34=%c1) args(%arg16=%alloc_1, %arg17=%alloc_2, %arg18=%arg118, %arg19=%arg119, %arg20=%arg120) : memref<128x8xf32, 1>, memref<8x128xf32, 1>, memref<8xf32, 2>, memref<8xf32, 2>, memref<64xf32, 2> {
            %c128_4 = arith.constant 128 : index
            %c0_5 = arith.constant 0 : index
            %c1_10 = arith.constant 1 : index
            %c16 = arith.constant 16 : index
            %c8_6 = arith.constant 8 : index
            %2 = affine.apply #map1()[%arg31]
            %3 = affine.apply #map1()[%arg32]
            scf.for %arg22 = %c0_5 to %c8_6 step %c1_10 {
              %c0_13 = arith.constant 0 : index
              %c1_14 = arith.constant 1 : index
              scf.for %arg23 = %c0_13 to %c8_6 step %c1_14 {
                  %6 = arith.remsi %arg31, %c16 : index
                  %8 = arith.muli %6, %c8_6 : index
                  %idx = arith.addi %8, %arg23 : index
                  %13 = memref.load %arg16[%idx, %arg22] : memref<128x8xf32, 1>
                  memref.store %13, %arg18[%arg23] : memref<8xf32, 2>
              }
              scf.for %arg23 = %c0_13 to %c8_6 step %c1_14 {
                %6 = arith.remsi %arg31, %c16 : index
                %7 = arith.divsi %arg31, %c16 : index
                %8 = arith.muli %7, %c8_6 : index
                %idx = arith.addi %8, %arg23 : index
                %13 = memref.load %arg17[%arg22, %idx] : memref<8x128xf32, 1>
                memref.store %13, %arg19[%arg23] : memref<8xf32, 2>
              }
              scf.for %yt = %c0_5 to %c8_6 step %c1_10 {
                scf.for %xt = %c0_5 to %c8_6 step %c1_10 {
                  %8 = arith.muli  %yt, %c8_6 : index
                  %idx = arith.addi %8, %xt : index
                  %10 = memref.load %arg18[%yt] : memref<8xf32, 2>
                  %11 = memref.load %arg19[%xt] : memref<8xf32, 2>
                  %12 = memref.load %arg20[%idx] : memref<64xf32, 2>
                  %13 = arith.mulf %10, %11 : f32
                  %14 = arith.addf %12, %13 : f32
                  memref.store %14, %arg20[%idx] : memref<64xf32, 2>
                }
              }

            }
            gpu.barrier
          }
          scf.for %yt = %c0 to %c8 step %c1 {
            scf.for %xt = %c0 to %c8 step %c1 {
              %8 = arith.muli %yt, %c8: index
              %x = arith.addi %xt, %8: index
              %12 = memref.load %arg120[%x] : memref<64xf32, 2>
              %18 = arith.remsi %tidx, %c16_0 : index
              %20 = arith.muli %18, %c8 : index
              %22 = arith.addi %0, %20 : index
              %index_x = arith.addi %22, %yt : index
              %19 = arith.divsi %tidx, %c16_0 : index

              %21 = arith.muli %19, %c8 : index
              %24 = arith.addi %1, %21 : index
              %index_y = arith.addi %24, %xt : index
              memref.store %12, %arg14[%index_x, %index_y] : memref<4096x4096xf32>
            }
          }
        }
      }
    }
    return
  }
}
