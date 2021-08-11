// RUN: air-opt %s -air-to-aie | FileCheck %s
// CHECK:    %[[VAL_0:.*]] = AIE.tile(2, 1)
// CHECK:         %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:         %[[VAL_2:.*]] = AIE.tile(0, 0)
// CHECK:         %[[VAL_4:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:         %[[VAL_3:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "buf0"} : memref<1024xi32, 2>

// CHECK:    AIE.mem(%[[VAL_2]])  {
// CHECK:           AIE.dmaStart(S2MM0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           AIE.useLock(%[[VAL_4]], Acquire, 0, 0)
// CHECK:           AIE.dmaBd(<%[[VAL_3]] : memref<1024xi32, 2>, 0, 0>, 0)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 1, 0)
// CHECK:           br ^bb1
// CHECK:         ^bb2:
// CHECK:           AIE.end
// CHECK:         }

// CHECK:    AIE.core(%[[VAL_2]])  {
// CHECK:           AIE.useLock(%[[VAL_4]], Acquire, 1, 0)
// CHECK:           AIE.useLock(%[[VAL_4]], Release, 0, 0)
// CHECK:           AIE.end
// CHECK:         }

// CHECK:           AIE.connect<South : 3, North : 0>
// CHECK:           AIE.connect<South : 7, North : 1>
// CHECK:           AIE.connect<North : 0, South : 2>
// CHECK:           AIE.connect<North : 1, South : 3>
// CHECK:         }
// CHECK:         AIE.flow(%[[VAL_0]], South : 0, %[[VAL_2]], DMA : 0)

// CHECK:     AIE.shimmux(%[[VAL_1]])  {
// CHECK:           AIE.connect<DMA : 0, South : 3>
// CHECK:           AIE.connect<DMA : 1, South : 7>
// CHECK:           AIE.connect<South : 2, DMA : 0>
// CHECK:           AIE.connect<South : 3, DMA : 1>
// CHECK:         }

module {

func @foo(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  %herd_cols = constant 1 : index
  %herd_rows = constant 1 : index
  air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { } {
    %c0 = constant 0 : index
    %c1024 = constant 0 : index
    %buf0 = memref.alloc() : memref<1024xi32, 2>
    air.dma_memcpy (%buf0, %ext0, [%c0], [%c0], %c1024) : (memref<1024xi32, 2>, memref<1024xi32>, [index], [index], index) -> ()
    memref.dealloc %buf0 : memref<1024xi32, 2>
    air.herd_terminator
  }
  return
}

}
