module @aie.partition_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(1, 2)
  %2 = AIE.tile(2, 0)
  %3 = AIE.tile(3, 0)
  %4 = AIE.objectFifo.createObjectFifo(%1, {%2}, 1) : !AIE.objectFifo<memref<32xi32, 2>>
  %5 = AIE.objectFifo.createObjectFifo(%2, {%1}, 1) : !AIE.objectFifo<memref<32xi32, 2>>
  %6 = AIE.objectFifo.createObjectFifo(%2, {%1}, 1) : !AIE.objectFifo<memref<32xi32, 2>>
  %7 = AIE.objectFifo.createObjectFifo(%0, {%1}, 1) : !AIE.objectFifo<memref<32xi32, 2>>
  %8 = AIE.objectFifo.createObjectFifo(%3, {%0}, 1) : !AIE.objectFifo<memref<32xi32, 2>>
  %9 = AIE.objectFifo.createObjectFifo(%3, {%0}, 1) : !AIE.objectFifo<memref<32xi32, 2>>
  %10 = AIE.core(%1) {
    affine.for %arg0 = 0 to 4096 step 32 {
      %12 = AIE.objectFifo.acquire<Consume> (%7 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %13 = AIE.objectFifo.subview.access %12[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      %14 = AIE.objectFifo.acquire<Consume> (%6 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %15 = AIE.objectFifo.subview.access %14[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      %16 = AIE.objectFifo.acquire<Consume> (%5 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %17 = AIE.objectFifo.subview.access %16[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      %18 = AIE.objectFifo.acquire<Produce> (%4 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %19 = AIE.objectFifo.subview.access %18[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      affine.for %arg1 = 0 to 32 {
        %20 = affine.load %15[%arg1] : memref<32xi32, 2>
        %21 = affine.load %17[%arg1] : memref<32xi32, 2>
        %22 = arith.addi %21, %20 : i32
        %23 = affine.load %13[%arg1] : memref<32xi32, 2>
        %24 = arith.addi %22, %23 : i32
        affine.store %24, %19[%arg1] : memref<32xi32, 2>
      }
      AIE.objectFifo.release<Produce> (%4 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
      AIE.objectFifo.release<Consume> (%5 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
      AIE.objectFifo.release<Consume> (%6 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
      AIE.objectFifo.release<Consume> (%7 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
    }
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %11 = AIE.core(%0) {
    affine.for %arg0 = 0 to 4096 step 32 {
      %12 = AIE.objectFifo.acquire<Consume> (%9 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %13 = AIE.objectFifo.subview.access %12[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      %14 = AIE.objectFifo.acquire<Consume> (%8 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %15 = AIE.objectFifo.subview.access %14[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      %16 = AIE.objectFifo.acquire<Produce> (%7 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
      %17 = AIE.objectFifo.subview.access %16[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
      affine.for %arg1 = 0 to 32 {
        %18 = affine.load %13[%arg1] : memref<32xi32, 2>
        %19 = affine.load %15[%arg1] : memref<32xi32, 2>
        %20 = arith.addi %19, %18 : i32
        affine.store %20, %17[%arg1] : memref<32xi32, 2>
      }
      AIE.objectFifo.release<Produce> (%7 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
      AIE.objectFifo.release<Consume> (%8 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
      AIE.objectFifo.release<Consume> (%9 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
    }
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}
