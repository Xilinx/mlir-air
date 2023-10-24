module {
  AIE.device(ipu) {
    AIE.shimDMAAllocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    AIE.shimDMAAllocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
    func.func @func0(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %c1_i32 = arith.constant 1 : i32
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c1_i32, %c64_i32] [%c0_i32, %c0_i32, %c0_i32]) {id = 2 : i32, metadata = @airMemcpyId2} : (i32, i32, memref<64xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c1_i32, %c64_i32] [%c0_i32, %c0_i32, %c0_i32]) {id = 7 : i32, metadata = @airMemcpyId7} : (i32, i32, memref<64xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      return
    }
  } {sym_name = "segment0"}
}


// -----
module {
  AIE.device(ipu) {
    AIE.shimDMAAllocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    AIE.shimDMAAllocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
    func.func @func0(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c64_i32 = arith.constant 64 : i32
      %c1_i32 = arith.constant 1 : i32
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c1_i32, %c64_i32] [%c0_i32, %c0_i32, %c0_i32]) {id = 2 : i32, metadata = @airMemcpyId2} : (i32, i32, memref<64xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg1[%c0_i32, %c0_i32, %c0_i32, %c0_i32] [%c1_i32, %c1_i32, %c1_i32, %c64_i32] [%c0_i32, %c0_i32, %c0_i32]) {id = 7 : i32, metadata = @airMemcpyId7} : (i32, i32, memref<64xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
      return
    }
  } {sym_name = "segment0"}
}

