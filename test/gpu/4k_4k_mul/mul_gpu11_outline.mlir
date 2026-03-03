module attributes {gpu.container_module} {
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
    return
  }
  func.func @main() {
    call @test_matmul() : () -> ()
    return
  }
  func.func @test_matmul() {
    %c4096_i64 = arith.constant 4096 : i64
    %c1_i32 = arith.constant 1 : i32
    %alloc = memref.alloc() : memref<4096x4096xf32>
    %alloc_0 = memref.alloc() : memref<4096x4096xf32>
    %alloc_1 = memref.alloc() : memref<4096x4096xf32>
    %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<4096x4096xf32> -> index
    %intptr_2 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<4096x4096xf32> -> index
    %0 = arith.index_cast %intptr : index to i64
    %1 = arith.index_cast %intptr_2 : index to i64
    %2 = llvm.inttoptr %0 : i64 to !llvm.ptr
    %3 = llvm.inttoptr %1 : i64 to !llvm.ptr
    llvm.call @mgpuInit(%2, %3, %c4096_i64, %c4096_i64) : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()
    %memref = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref, %alloc : memref<4096x4096xf32>, memref<4096x4096xf32>
    %memref_3 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_3, %alloc_0 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %memref_4 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_4, %alloc_1 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %4 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %5 = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    %6 = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    llvm.call @mgpuEventRecord(%5, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    call @forward(%memref, %memref_3, %memref_4) : (memref<4096x4096xf32>, memref<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    llvm.call @mgpuEventRecord(%6, %4) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @mgpuEventSynchronize(%6) : (!llvm.ptr) -> ()
    %7 = llvm.alloca %c1_i32 x f32 : (i32) -> !llvm.ptr
    %8 = llvm.call @mgpuEventElapsedTime(%7, %5, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.call @mgpuStreamDestroy(%4) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%5) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%6) : (!llvm.ptr) -> ()
    gpu.memcpy  %alloc_1, %memref_4 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %intptr_5 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<4096x4096xf32> -> index
    %intptr_6 = memref.extract_aligned_pointer_as_index %alloc : memref<4096x4096xf32> -> index
    %intptr_7 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<4096x4096xf32> -> index
    %9 = arith.index_cast %intptr_5 : index to i64
    %10 = arith.index_cast %intptr_6 : index to i64
    %11 = arith.index_cast %intptr_7 : index to i64
    %12 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %13 = llvm.inttoptr %10 : i64 to !llvm.ptr
    %14 = llvm.inttoptr %11 : i64 to !llvm.ptr
    llvm.call @mgpuCheckOutput(%12, %13, %14, %c4096_i64, %c4096_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    return
  }
  func.func @forward(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    gpu.launch_func  @forward_module::@forward_module blocks in (%c32, %c32, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<4096x4096xf32>, %arg1 : memref<4096x4096xf32>, %arg2 : memref<4096x4096xf32>)
    return
  }
  gpu.module @forward_module {
    gpu.func @forward_module(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) workgroup(%arg3 : memref<128x8xf32, 3>, %arg4 : memref<8x128xf32, 3>) private(%arg5 : memref<8xf32, 5>, %arg6 : memref<8xf32, 5>, %arg7 : memref<64xf32, 5>) kernel attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 32, 32, 1>} {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %0 = arith.remsi %thread_id_x_0, %c128 : index
      %1 = arith.divsi %thread_id_x_0, %c128 : index
      %2 = arith.muli %block_id_y, %c128 overflow<nsw> : index
      %3 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
      %5 = arith.cmpi slt, %4, %c64 : index
      cf.cond_br %5, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      memref.store %cst, %arg7[%4] : memref<64xf32, 5>
      %6 = arith.addi %4, %c1 : index
      cf.br ^bb1(%6 : index)
    ^bb3:  // pred: ^bb1
      cf.br ^bb4(%c0 : index)
    ^bb4(%7: index):  // 2 preds: ^bb3, ^bb32
      %8 = arith.cmpi slt, %7, %c4096 : index
      cf.cond_br %8, ^bb5, ^bb33
    ^bb5:  // pred: ^bb4
      cf.br ^bb6(%c0 : index)
    ^bb6(%9: index):  // 2 preds: ^bb5, ^bb7
      %10 = arith.cmpi slt, %9, %c4 : index
      cf.cond_br %10, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %11 = arith.addi %2, %0 : index
      %12 = arith.muli %1, %c4 : index
      %13 = arith.addi %12, %9 : index
      %14 = arith.addi %13, %7 : index
      %15 = arith.remsi %11, %c128 : index
      %16 = arith.remsi %14, %c8 : index
      %17 = memref.load %arg0[%11, %14] : memref<4096x4096xf32>
      memref.store %17, %arg3[%15, %16] : memref<128x8xf32, 3>
      %18 = arith.addi %9, %c1 : index
      cf.br ^bb6(%18 : index)
    ^bb8:  // pred: ^bb6
      cf.br ^bb9(%c0 : index)
    ^bb9(%19: index):  // 2 preds: ^bb8, ^bb10
      %20 = arith.cmpi slt, %19, %c4 : index
      cf.cond_br %20, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %21 = arith.addi %3, %0 : index
      %22 = arith.muli %1, %c4 : index
      %23 = arith.addi %22, %19 : index
      %24 = arith.addi %23, %7 : index
      %25 = arith.remsi %21, %c128 : index
      %26 = arith.remsi %24, %c8 : index
      %27 = memref.load %arg1[%24, %21] : memref<4096x4096xf32>
      memref.store %27, %arg4[%26, %25] : memref<8x128xf32, 3>
      %28 = arith.addi %19, %c1 : index
      cf.br ^bb9(%28 : index)
    ^bb11:  // pred: ^bb9
      gpu.barrier
      cf.br ^bb12(%c0 : index)
    ^bb12(%29: index):  // 2 preds: ^bb11, ^bb25
      %30 = arith.cmpi slt, %29, %c8 : index
      cf.cond_br %30, ^bb13, ^bb26
    ^bb13:  // pred: ^bb12
      cf.br ^bb14(%c0 : index)
    ^bb14(%31: index):  // 2 preds: ^bb13, ^bb15
      %32 = arith.cmpi slt, %31, %c8 : index
      cf.cond_br %32, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      %33 = arith.remsi %thread_id_x, %c16 : index
      %34 = arith.muli %33, %c8 : index
      %35 = arith.addi %34, %31 : index
      %36 = memref.load %arg3[%35, %29] : memref<128x8xf32, 3>
      memref.store %36, %arg5[%31] : memref<8xf32, 5>
      %37 = arith.addi %31, %c1 : index
      cf.br ^bb14(%37 : index)
    ^bb16:  // pred: ^bb14
      cf.br ^bb17(%c0 : index)
    ^bb17(%38: index):  // 2 preds: ^bb16, ^bb18
      %39 = arith.cmpi slt, %38, %c8 : index
      cf.cond_br %39, ^bb18, ^bb19
    ^bb18:  // pred: ^bb17
      %40 = arith.divsi %thread_id_x, %c16 : index
      %41 = arith.muli %40, %c8 : index
      %42 = arith.addi %41, %38 : index
      %43 = memref.load %arg4[%29, %42] : memref<8x128xf32, 3>
      memref.store %43, %arg6[%38] : memref<8xf32, 5>
      %44 = arith.addi %38, %c1 : index
      cf.br ^bb17(%44 : index)
    ^bb19:  // pred: ^bb17
      cf.br ^bb20(%c0 : index)
    ^bb20(%45: index):  // 2 preds: ^bb19, ^bb24
      %46 = arith.cmpi slt, %45, %c8 : index
      cf.cond_br %46, ^bb21, ^bb25
    ^bb21:  // pred: ^bb20
      cf.br ^bb22(%c0 : index)
    ^bb22(%47: index):  // 2 preds: ^bb21, ^bb23
      %48 = arith.cmpi slt, %47, %c8 : index
      cf.cond_br %48, ^bb23, ^bb24
    ^bb23:  // pred: ^bb22
      %49 = arith.muli %45, %c8 : index
      %50 = arith.addi %49, %47 : index
      %51 = memref.load %arg5[%45] : memref<8xf32, 5>
      %52 = memref.load %arg6[%47] : memref<8xf32, 5>
      %53 = memref.load %arg7[%50] : memref<64xf32, 5>
      %54 = arith.mulf %51, %52 : f32
      %55 = arith.addf %53, %54 : f32
      memref.store %55, %arg7[%50] : memref<64xf32, 5>
      %56 = arith.addi %47, %c1 : index
      cf.br ^bb22(%56 : index)
    ^bb24:  // pred: ^bb22
      %57 = arith.addi %45, %c1 : index
      cf.br ^bb20(%57 : index)
    ^bb25:  // pred: ^bb20
      %58 = arith.addi %29, %c1 : index
      cf.br ^bb12(%58 : index)
    ^bb26:  // pred: ^bb12
      gpu.barrier
      cf.br ^bb27(%c0 : index)
    ^bb27(%59: index):  // 2 preds: ^bb26, ^bb31
      %60 = arith.cmpi slt, %59, %c8 : index
      cf.cond_br %60, ^bb28, ^bb32
    ^bb28:  // pred: ^bb27
      cf.br ^bb29(%c0 : index)
    ^bb29(%61: index):  // 2 preds: ^bb28, ^bb30
      %62 = arith.cmpi slt, %61, %c8 : index
      cf.cond_br %62, ^bb30, ^bb31
    ^bb30:  // pred: ^bb29
      %63 = arith.muli %59, %c8 : index
      %64 = arith.addi %61, %63 : index
      %65 = memref.load %arg7[%64] : memref<64xf32, 5>
      %66 = arith.remsi %thread_id_x_0, %c16 : index
      %67 = arith.muli %66, %c8 : index
      %68 = arith.addi %2, %67 : index
      %69 = arith.addi %68, %59 : index
      %70 = arith.divsi %thread_id_x_0, %c16 : index
      %71 = arith.muli %70, %c8 : index
      %72 = arith.addi %3, %71 : index
      %73 = arith.addi %72, %61 : index
      memref.store %65, %arg2[%69, %73] : memref<4096x4096xf32>
      %74 = arith.addi %61, %c1 : index
      cf.br ^bb29(%74 : index)
    ^bb31:  // pred: ^bb29
      %75 = arith.addi %59, %c1 : index
      cf.br ^bb27(%75 : index)
    ^bb32:  // pred: ^bb27
      %76 = arith.addi %7, %c8 : index
      cf.br ^bb4(%76 : index)
    ^bb33:  // pred: ^bb4
      gpu.return
    }
  }
}

