//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_PASSES
#define AIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ParallelToHerd : Pass<"air-par-to-herd", "ModuleOp"> {
  let summary = "Lower scf.parallel to air.launch_herd";
  let constructor = "xilinx::air::createParallelToHerdPass()";
  let description =  [{
    Lower scf.parallel to air.launch_herd
  }];
  let options = [
    Option<"clAssignDepth", "depth", "int",
          /*default=*/"-1",
          "Given a nest of parallel for loops, which depth to map to herd launch">
  ];
}

def ParallelToLaunch : Pass<"air-par-to-launch", "ModuleOp"> {
  let summary = "Lower scf.parallel to air.launch";
  let constructor = "xilinx::air::createParallelToLaunchPass()";
  let description =  [{
    Lower scf.parallel to air.launch
  }];
  let options = [
    Option<"clAssignDepth", "depth", "int",
          /*default=*/"-1",
          "Given a nest of parallel for loops, which depth to map to air.launch">,
    Option<"clHasPartition", "has-air-partition", "bool", /*default=*/"false",
           "Whether to create an air.partition op in air.launch's region">,
  ];
}

def CopyToDma : Pass<"air-copy-to-dma", "ModuleOp"> {
  let summary = "Convert memcpy to air.dma_memcpy_nd";
  let constructor = "xilinx::air::createCopyToDmaPass()";
  let description =  [{
    Convert memcpy to air.dma_memcpy_nd
  }];
}

def DmaToChannel : Pass<"air-dma-to-channel", "ModuleOp"> {
  let summary = "Convert air.dma_memcpy_nd to air.channel";
  let constructor = "xilinx::air::createDmaToChannelPass()";
  let description =  [{
    Convert air.dma_memcpy_nd to air.channel
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
  let options = [
    Option<"clLoweringType", "lowering-type", "std::string", 
          /*default=*/"\"buffers\"",
          "Type of lowering to use for core-to-core communication. Can be 'buffer' or 'getput'">
  ];
}

#endif
