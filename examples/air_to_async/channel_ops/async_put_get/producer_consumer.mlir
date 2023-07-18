module {
  memref.global "private" @channel_0 : memref<i64> = dense<0>
  func.func @forward(%arg0: memref<32x32x32xi32>) attributes {llvm.emit_c_interface} { 
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    // producer
    %token_0 = async.execute {
      scf.for %ii = %c0 to %c32 step %c1 {
        %alloc = memref.alloc() : memref<32x32xi32>
        %val = arith.index_cast %ii : index to i32
        linalg.fill ins(%val : i32) outs(%alloc : memref<32x32xi32>)
        // put %alloc into channel_0
        %0 = memref.get_global @channel_0 : memref<i64>
        %1 = builtin.unrealized_conversion_cast %0 : memref<i64> to memref<i64>
        %2 = builtin.unrealized_conversion_cast %alloc : memref<32x32xi32> to memref<?x?xi32>
        func.call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%1, %2, %c0, %c0, %c32, %c32, %c1, %c32) : (memref<i64>,  memref<?x?xi32>, index, index, index, index, index, index) -> ()
        memref.dealloc %alloc : memref<32x32xi32>
        scf.yield
      }
      async.yield
    }

    // consumer
    %token_1 = async.execute {
      scf.for %ii = %c0 to %c32 step %c1 {
        %alloc = memref.alloc() : memref<32x32xi32>
        // get %alloc from channel_0
        %0 = memref.get_global @channel_0 : memref<i64>
        %1 = builtin.unrealized_conversion_cast %0 : memref<i64> to memref<i64>
        %2 = builtin.unrealized_conversion_cast %alloc : memref<32x32xi32> to memref<?x?xi32>
        func.call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%1, %2, %c0, %c0, %c32, %c32, %c1, %c32) : (memref<i64>,  memref<?x?xi32>, index, index, index, index, index, index) -> ()
        // copy %alloc to %arg0[ii]
        scf.for %arg2 = %c0 to %c32 step %c1 {
          scf.for %arg3 = %c0 to %c32 step %c1 {
            %3 = memref.load %alloc[%arg2, %arg3] : memref<32x32xi32>
            memref.store %3, %arg0[%ii, %arg2, %arg3] : memref<32x32x32xi32>
            scf.yield
          }
          scf.yield
        }
        memref.dealloc %alloc : memref<32x32xi32>
        scf.yield
      }
      async.yield
    }

    async.await %token_0 : !async.token
    async.await %token_1 : !async.token
    
    return
  }
  func.func private @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
  func.func private @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
}

