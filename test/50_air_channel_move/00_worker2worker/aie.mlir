module @aie.partition_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(1, 2)
  %2 = AIE.objectFifo.createObjectFifo(%0, {%1}, 2) : !AIE.objectFifo<memref<32xi32, 2>>
  %3 = AIE.core(%1) {
    %5 = AIE.objectFifo.acquire<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
    %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
    AIE.objectFifo.release<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %4 = AIE.core(%0) {
    %5 = AIE.objectFifo.acquire<Produce> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
    %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
    AIE.objectFifo.release<Produce> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}

