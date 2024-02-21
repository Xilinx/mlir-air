module @aie.partition_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(2, 0)
  %2 = AIE.objectFifo.createObjectFifo(%1, {%0}, 1) : !AIE.objectFifo<memref<32xi32, 2>>
  %3 = AIE.objectFifo.createObjectFifo(%0, {%1}, 1) : !AIE.objectFifo<memref<1xi32, 2>>
  %4 = AIE.core(%0) {
    %5 = AIE.objectFifo.acquire<Produce> (%3 : !AIE.objectFifo<memref<1xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<1xi32, 2>>
    %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<1xi32, 2>> -> memref<1xi32, 2>
    affine.for %arg0 = 0 to 4096 step 32 {
      %7 = AIE.objectFifo.acquire<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %8 = AIE.objectFifo.subview.access %7[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      affine.for %arg1 = 0 to 32 {
        %9 = affine.load %8[%arg1] : memref<32xi32, 2>
        %10 = affine.load %6[0] : memref<1xi32, 2>
        %11 = arith.addi %9, %10 : i32
        affine.store %11, %6[0] : memref<1xi32, 2>
      }
      AIE.objectFifo.release<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
    }
    AIE.objectFifo.release<Produce> (%3 : !AIE.objectFifo<memref<1xi32, 2>>, 1)
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}

