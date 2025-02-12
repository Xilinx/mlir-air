module @aie.partition_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(2, 0)
  %2 = AIE.objectFifo.createObjectFifo(%1, {%0}, 2) {sym_name = "inA_0_0"} : !AIE.objectFifo<memref<32xi32, 2>>
  %3 = AIE.objectFifo.createObjectFifo(%1, {%0}, 2) {sym_name = "inB_0_0"} : !AIE.objectFifo<memref<32xi32, 2>>
  %4 = AIE.objectFifo.createObjectFifo(%0, {%1}, 2) {sym_name = "outC_0_0"} : !AIE.objectFifo<memref<32xi32, 2>>
  %5 = AIE.core(%0) {
    affine.for %arg0 = 0 to 4096 step 32 {
      %6 = AIE.objectFifo.acquire<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %7 = AIE.objectFifo.subview.access %6[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      %8 = AIE.objectFifo.acquire<Consume> (%3 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %9 = AIE.objectFifo.subview.access %8[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      %10 = AIE.objectFifo.acquire<Produce> (%4 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      affine.for %arg1 = 0 to 32 {
        %12 = affine.load %7[%arg1] : memref<32xi32, 2>
        %13 = affine.load %9[%arg1] : memref<32xi32, 2>
        %14 = arith.addi %13, %12 : i32
        affine.store %14, %11[%arg1] : memref<32xi32, 2>
      }
      AIE.objectFifo.release<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
      AIE.objectFifo.release<Consume> (%3 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
      AIE.objectFifo.release<Produce> (%4 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
    }
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}

