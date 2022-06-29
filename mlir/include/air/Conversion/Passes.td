#ifndef AIR_CONVERSION_PASSES
#define AIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AffineToAIR : Pass<"affine-to-air", "ModuleOp"> {
  let summary = "Affine to AIR";
  let constructor = "xilinx::air::createAffineToAIRPass()";
  let description = [{
    This pass converts affine.parallel and scf.parallel operations into AIR
    dialect herd launch operations.

    It also converts affine.dma_start and linalg.copy operations into AIR
    DMA operations.

    ### Example - convert scf + linalg

    ### Input
    ```mlir
    #map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
    module  {
      func.func @myFunc(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> memref<64x64xf32> {
        %c64 = constant 64 : index
        %c16 = constant 16 : index
        %c0 = constant 0 : index
        %0 = memref.alloc() : memref<64x64xf32>
        scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c64) step (%c16, %c16) {
          %1 = memref.subview %arg0[%arg2, 0] [16, 64] [1, 1] : memref<64x64xf32> to memref<16x64xf32, #map>
          %2 = memref.subview %arg1[0, %arg3] [64, 16] [1, 1] : memref<64x64xf32> to memref<64x16xf32, #map>
          %3 = memref.subview %0[%arg2, %arg3] [16, 16] [1, 1] : memref<64x64xf32> to memref<16x16xf32, #map>
          %4 = memref.alloc() : memref<16x64xf32, 2>
          %5 = memref.alloc() : memref<64x16xf32, 2>
          %6 = memref.alloc() : memref<16x16xf32, 2>
          linalg.copy(%1, %4) : memref<16x64xf32, #map>, memref<16x64xf32, 2>
          linalg.copy(%2, %5) : memref<64x16xf32, #map>, memref<64x16xf32, 2>
          linalg.copy(%3, %6) : memref<16x16xf32, #map>, memref<16x16xf32, 2>
          linalg.matmul ins(%4, %5 : memref<16x64xf32, 2>, memref<64x16xf32, 2>) outs(%6 : memref<16x16xf32, 2>)
          linalg.copy(%6, %3) : memref<16x16xf32, 2>, memref<16x16xf32, #map>
          memref.dealloc %4 : memref<16x64xf32, 2>
          memref.dealloc %5 : memref<64x16xf32, 2>
          memref.dealloc %6 : memref<16x16xf32, 2>
          scf.yield
        }
        return %0 : memref<64x64xf32>
      }
    }
    ```
    ### Output
    ```mlir
    module  {
      func.func @myFunc(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> memref<64x64xf32> {
        %c4 = constant 4 : index
        %0 = memref.alloc() : memref<64x64xf32>
        air.launch_herd tile (%arg2, %arg3) in (%arg4=%c4, %arg5=%c4) args(%arg6=%arg0, %arg7=%arg1, %arg8=%0) : memref<64x64xf32>,memref<64x64xf32>,memref<64x64xf32>attributes {sym_name = "herd_0"} {
          %c16 = constant 16 : index
          %1 = muli %arg2, %c16 : index
          %2 = muli %arg3, %c16 : index
          %3 = memref.alloc() : memref<16x64xf32, 2>
          %4 = memref.alloc() : memref<64x16xf32, 2>
          %5 = memref.alloc() : memref<16x16xf32, 2>
          %c0 = constant 0 : index
          %c64 = constant 64 : index
          %c1024 = constant 1024 : index
          air.dma_memcpy_2d (%3, %arg6, [%c0, %c0], [%1, %c0], %c1024, %c64, %c64) {id = 1 : i32} : (memref<16x64xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
          air.dma_memcpy_2d (%4, %arg7, [%c0, %c0], [%c0, %2], %c1024, %c64, %c16) {id = 2 : i32} : (memref<64x16xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
          %c256 = constant 256 : index
          air.dma_memcpy_2d (%5, %arg8, [%c0, %c0], [%1, %2], %c256, %c64, %c16) {id = 3 : i32} : (memref<16x16xf32, 2>, memref<64x64xf32>, [index, index], [index, index], index, index, index) -> ()
          linalg.matmul ins(%3, %4 : memref<16x64xf32, 2>, memref<64x16xf32, 2>) outs(%5 : memref<16x16xf32, 2>)
          air.dma_memcpy_2d (%arg8, %5, [%1, %2], [%c0, %c0], %c256, %c64, %c16) {id = 4 : i32} : (memref<64x64xf32>, memref<16x16xf32, 2>, [index, index], [index, index], index, index, index) -> ()
          memref.dealloc %3 : memref<16x64xf32, 2>
          memref.dealloc %4 : memref<64x16xf32, 2>
          memref.dealloc %5 : memref<16x16xf32, 2>
          air.herd_terminator
        }
        return %0 : memref<64x64xf32>
      }
    }
    ```
  }];
}

def AIRToCpu : Pass<"air-to-cpu", "ModuleOp"> {
  let summary = "AIR dialect lowering";
  let constructor = "xilinx::air::createAIRToCpuPass()";
  let description = [{
  }];
}

def AIRLowering : Pass<"air-to-std", "ModuleOp"> {
  let summary = "AIR dialect lowering";
  let constructor = "xilinx::air::createAIRLoweringPass()";
  let description = [{
    This pass converts AIR dialect herd launch operations into loop nests
    representing the host-side control program for the herd. It also converts
    AIR dialect memcpy operations into AIRRt memcpy operations.

    ### Example -  - A 1x1 herd copying a 1024xi32 vector from L3 memory into an L1 buffer

    ### Input

    ```mlir
      func.func @foo(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
        %herd_cols = constant 1 : index
        %herd_rows = constant 1 : index
        air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="foo"} {
          %c0 = constant 0 : index
          %c1024 = constant 0 : index
          %buf0 = memref.alloc() : memref<1024xi32, 2>
          air.dma_memcpy (%buf0, %ext0, [%c0], [%c0], %c1024) {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>, [index], [index], index) -> ()
          memref.dealloc %buf0 : memref<1024xi32, 2>
          air.herd_terminator
        }
        return
      }
    ```

    #### Output

    ```mlir
      func.func @foo(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
        %c1 = constant 1 : index
        %0 = airrt.herd_load "foo"() : () -> i32
        affine.for %arg2 = 0 to 1 {
          affine.for %arg3 = 0 to 1 {
            %c0 = constant 0 : index
            %c1024 = constant 1024 : index
            %1 = memref.alloc() : memref<1024xi32, 2>
            %c1_i32 = constant 1 : i32
            %2 = index_cast %arg3 : index to i64
            %3 = index_cast %arg2 : index to i64
            %4 = index_cast %c0 : index to i64
            %5 = index_cast %c1024 : index to i64
            airrt.dma_memcpy(%c1_i32, %2, %3, %arg0[%4], %5) : (i32, i64, i64, memref<1024xi32>, [i64], i64) -> ()
            memref.dealloc %1 : memref<1024xi32, 2>
          } {air.herd_launch = "inner"}
        } {air.herd_launch = "outer"}
        return
      }
    ```
  }];
}

def AIRToAIE : Pass<"air-to-aie", "ModuleOp"> {
  let summary = "Lower air.launch_herd to AIE dialect";
  let constructor = "xilinx::air::createAIRToAIEPass()";
  let description = [{
    This pass lowers AIR dialect herd_launch operations into AIE dialect
    dialect and AIRRt dialect.

    AIE.core operations are generated for each tile in the herd. The AIR DMA
    operations performed by each tile are lowered to AIE.mem operations to
    perform the transfers and AIE.locks are allocated to synchronize between
    the cores and the tile DMAs. AIE.buffers are allocated for any L1 memrefs
    allocated in the AIR program. AIE.flow operations are generated to 
    connect the sending and receiving DMAs.

    ### Example - A 1x1 herd copying a 1024xi32 vector from L3 memory into an L1 buffer

    ### Input

    ```mlir
      func.func @foo(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
        %herd_cols = constant 1 : index
        %herd_rows = constant 1 : index
        air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="foo"} {
          %c0 = constant 0 : index
          %c1024 = constant 0 : index
          %buf0 = memref.alloc() : memref<1024xi32, 2>
          air.dma_memcpy (%buf0, %ext0, [%c0], [%c0], %c1024) {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>, [index], [index], index) -> ()
          memref.dealloc %buf0 : memref<1024xi32, 2>
          air.herd_terminator
        }
        return
      }
    ```

    #### Output
    
    The AIE resource allocation,
    ```mlir
      module @aie.foo  {
        ...
        %3 = AIE.tile(3, 3)
        %4 = AIE.lock(%3, 0)
        %5 = AIE.buffer(%3) {sym_name = "buf0"} : memref<1024xi32, 2>
        ...
        AIE.flow(%0, South : 0, %3, DMA : 0)
    ```
    the AIE DMA program,
    ```mlir
      %6 = AIE.mem(%3)  {
        %10 = AIE.dmaStart(S2MM0, ^bb1, ^bb2)
      ^bb1:  // 2 preds: ^bb0, ^bb1
        AIE.useLock(%4, Acquire, 0, 0)
        AIE.dmaBd(<%5 : memref<1024xi32, 2>, 0, 0>, 0)
        AIE.useLock(%4, Release, 1, 0)
        br ^bb1
      ^bb2:  // pred: ^bb0
        AIE.end
      }
    ```
    and the AIE Core program.
    ```mlir
      %7 = AIE.core(%3)  {
        br ^bb1
      ^bb1:  // pred: ^bb0
        br ^bb2
      ^bb2:  // pred: ^bb1
        AIE.useLock(%4, Acquire, 1, 0)
        AIE.useLock(%4, Release, 0, 0)
        AIE.end
      }
    }
    ```

    The output of the pass will include the static metadata
    containing allocation information. In this case it specifies the
    allocation of channel 2 of the shim DMA in column 2 of the AIE
    array to perform the copy with id=1 in tile 0,0 
    ```mlir
      airrt.module_metadata {
        airrt.herd_metadata {dma_allocations = [{channel = 2 : i64, col = 0 : i64, id = 1 : i64, location = 2 : i64, row = 0 : i64 ...
      }
    ```
    The control program itself will be generated from this output by the 
    air-to-std pass.

  }
  }];
}

def AIRRtToLLVM : Pass<"airrt-to-llvm", "ModuleOp"> {
  let summary = "Lower AIRRt dialect to LLVM dialect";
  let constructor = "xilinx::airrt::createAIRRtToLLVMPass()";
  let description =  [{
    This pass lowers AIRRt dialect to function calls and data structures
    matching those found in air_host.h.

    AIRRt static metadata is transformed to LLVM dialect data structures.
    The data is generated as a number of globals with external linkage.
    The data layout is closely tied the AIR runtime and the definitions in
    air_host.h.  Any changes to this pass must be reflected there.
  }];
}

def AIRPipelineToAffine : Pass<"air-pipeline-to-affine", "ModuleOp"> {
  let summary = "Lower air.pipeline stages to affine.if";
  let constructor = "xilinx::air::createAIRPipelineToAffinePass()";
  let description =  [{
    Lower air.pipeline stages to affine.if
  }];
}

#endif
