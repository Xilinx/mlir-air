module attributes {torch.debug_module_name = "model"} {
  memref.global "private" @channel_7 : memref<i64> = dense<0>
  memref.global "private" @channel_6 : memref<i64> = dense<0>
  memref.global "private" @channel_5 : memref<i64> = dense<0>
  memref.global "private" @channel_4 : memref<i64> = dense<0>
  memref.global "private" @channel_3 : memref<i64> = dense<0>
  memref.global "private" @channel_2 : memref<i64> = dense<0>
  memref.global "private" @channel_1 : memref<i64> = dense<0>
  memref.global "private" @channel_0 : memref<i64> = dense<0>
  func.func @forward(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>, %arg2: memref<32x32xi32>, %arg3: memref<32x32xi32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<32x32xi32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    memref.copy %alloc, %alloc_0 : memref<32x32xi32> to memref<32x32xi32>
    %0 = memref.get_global @channel_0 : memref<i64>
    %1 = builtin.unrealized_conversion_cast %0 : memref<i64> to memref<i64>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<32x32xi32> to memref<?x?xi32>
    // put %arg0 into channel_0
    call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%1, %2, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %3 = memref.get_global @channel_1 : memref<i64>
    %4 = builtin.unrealized_conversion_cast %3 : memref<i64> to memref<i64>
    %5 = builtin.unrealized_conversion_cast %arg1 : memref<32x32xi32> to memref<?x?xi32>
    // put %arg1 into channel_1
    call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%4, %5, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %6 = memref.get_global @channel_2 : memref<i64>
    %7 = builtin.unrealized_conversion_cast %6 : memref<i64> to memref<i64>
    %8 = builtin.unrealized_conversion_cast %alloc_0 : memref<32x32xi32> to memref<?x?xi32>
    // put %alloc_0 into channel_2 
    call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%7, %8,%c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %token = async.execute {
      %alloc_2 = memref.alloc() : memref<32x32xi32>
      %alloc_3 = memref.alloc() : memref<32x32xi32>
      %alloc_4 = memref.alloc() : memref<32x32xi32>
      %24 = memref.get_global @channel_0 : memref<i64>
      %25 = builtin.unrealized_conversion_cast %24 : memref<i64> to memref<i64>
      %26 = builtin.unrealized_conversion_cast %alloc_2 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%25, %26, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      %27 = memref.get_global @channel_1 : memref<i64>
      %28 = builtin.unrealized_conversion_cast %27 : memref<i64> to memref<i64>
      %29 = builtin.unrealized_conversion_cast %alloc_3 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%28, %29, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      %30 = memref.get_global @channel_2 : memref<i64>
      %31 = builtin.unrealized_conversion_cast %30 : memref<i64> to memref<i64>
      %32 = builtin.unrealized_conversion_cast %alloc_4 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%31, %32, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      linalg.matmul ins(%alloc_2, %alloc_3 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_4 : memref<32x32xi32>)
      %33 = memref.get_global @channel_3 : memref<i64>
      %34 = builtin.unrealized_conversion_cast %33 : memref<i64> to memref<i64>
      %35 = builtin.unrealized_conversion_cast %alloc_4 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%34, %35, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      memref.dealloc %alloc_2 : memref<32x32xi32>
      memref.dealloc %alloc_3 : memref<32x32xi32>
      memref.dealloc %alloc_4 : memref<32x32xi32>
      async.yield
    }
    async.await %token : !async.token
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32> // result of second mm
    memref.copy %alloc, %alloc_1 : memref<32x32xi32> to memref<32x32xi32> // zero init

    // get %alloc_0 from channel_3
    %chn3 = memref.get_global @channel_3 : memref<i64>
    %chn3_cast = builtin.unrealized_conversion_cast %chn3 : memref<i64> to memref<i64>
    %alloc_0_cast = builtin.unrealized_conversion_cast %alloc_0 : memref<32x32xi32> to memref<?x?xi32>
    call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%chn3_cast, %alloc_0_cast, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()

    %12 = memref.get_global @channel_4 : memref<i64>
    %13 = builtin.unrealized_conversion_cast %12 : memref<i64> to memref<i64>
    %14 = builtin.unrealized_conversion_cast %alloc_0 : memref<32x32xi32> to memref<?x?xi32>
    // put %alloc_0 into channel_4
    call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%13, %14, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %15 = memref.get_global @channel_5 : memref<i64>
    %16 = builtin.unrealized_conversion_cast %15 : memref<i64> to memref<i64>
    %17 = builtin.unrealized_conversion_cast %arg2 : memref<32x32xi32> to memref<?x?xi32>
    // put %arg2 into channel_5
    call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%16, %17,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %18 = memref.get_global @channel_6 : memref<i64>
    %19 = builtin.unrealized_conversion_cast %18 : memref<i64> to memref<i64>
    %20 = builtin.unrealized_conversion_cast %alloc_1 : memref<32x32xi32> to memref<?x?xi32>
    // put %alloc_1 into channel_6
    call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%19, %20,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %token_0 = async.execute {
      %alloc_2 = memref.alloc() : memref<32x32xi32>
      %alloc_3 = memref.alloc() : memref<32x32xi32>
      %alloc_4 = memref.alloc() : memref<32x32xi32>
      %24 = memref.get_global @channel_4 : memref<i64>
      %25 = builtin.unrealized_conversion_cast %24 : memref<i64> to memref<i64>
      %26 = builtin.unrealized_conversion_cast %alloc_2 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%25, %26, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      %27 = memref.get_global @channel_5 : memref<i64>
      %28 = builtin.unrealized_conversion_cast %27 : memref<i64> to memref<i64>
      %29 = builtin.unrealized_conversion_cast %alloc_3 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%28,  %29, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      %30 = memref.get_global @channel_6 : memref<i64>
      %31 = builtin.unrealized_conversion_cast %30 : memref<i64> to memref<i64>
      %32 = builtin.unrealized_conversion_cast %alloc_4 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%31,  %32,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      linalg.matmul ins(%alloc_2, %alloc_3 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_4 : memref<32x32xi32>)
      %33 = memref.get_global @channel_7 : memref<i64>
      %34 = builtin.unrealized_conversion_cast %33 : memref<i64> to memref<i64>
      %35 = builtin.unrealized_conversion_cast %alloc_4 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%34, %35,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      memref.dealloc %alloc_2 : memref<32x32xi32>
      memref.dealloc %alloc_3 : memref<32x32xi32>
      memref.dealloc %alloc_4 : memref<32x32xi32>
      async.yield
    }
    async.await %token_0 : !async.token
    // get %alloc_1 from channel_7
    %chn7 = memref.get_global @channel_7 : memref<i64>
    %chn7_cast = builtin.unrealized_conversion_cast %chn7 : memref<i64> to memref<i64>
    %alloc_1_cast = builtin.unrealized_conversion_cast %alloc_1 : memref<32x32xi32> to memref<?x?xi32>
    call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%chn7_cast, %alloc_1_cast,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    memref.copy %alloc_1, %arg3 : memref<32x32xi32> to memref<32x32xi32>
    return
  }
  func.func private @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
  func.func private @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
}

