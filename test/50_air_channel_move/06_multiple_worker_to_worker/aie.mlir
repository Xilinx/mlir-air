module @aie.partition_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(2, 1)
  %2 = AIE.tile(1, 2)
  %3 = AIE.tile(2, 2)
  %4 = AIE.objectFifo.createObjectFifo(%0, {%1}, 1) : !AIE.objectFifo<memref<32xi32>>
  %5 = AIE.objectFifo.createObjectFifo(%2, {%3}, 1) : !AIE.objectFifo<memref<32xi32>>
  %6 = AIE.core(%3) {
    %10 = AIE.objectFifo.acquire<Consume> (%5 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
    %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
    AIE.objectFifo.release<Consume> (%5 : !AIE.objectFifo<memref<32xi32>>, 1)
    AIE.end
  } {elf_file = "partition_0_core_2_2.elf"}
  %7 = AIE.core(%2) {
    %10 = AIE.objectFifo.acquire<Produce> (%5 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
    %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
    AIE.objectFifo.release<Produce> (%5 : !AIE.objectFifo<memref<32xi32>>, 1)
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %8 = AIE.core(%1) {
    %10 = AIE.objectFifo.acquire<Consume> (%4 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
    %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
    AIE.objectFifo.release<Consume> (%4 : !AIE.objectFifo<memref<32xi32>>, 1)
    AIE.end
  } {elf_file = "partition_0_core_2_1.elf"}
  %9 = AIE.core(%0) {
    %10 = AIE.objectFifo.acquire<Produce> (%4 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
    %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
    AIE.objectFifo.release<Produce> (%4 : !AIE.objectFifo<memref<32xi32>>, 1)
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}