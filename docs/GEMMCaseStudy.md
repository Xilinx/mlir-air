# Bfloat16 GEMM on Ryzen AI: A Case Study for MLIR-AIR compilation pipeline.

## MLIR-AIR Compilation Recipe

The MLIR-AIR compilation pipeline used by the Ryzen AI E2E [board test](https://github.com/Xilinx/mlir-air/blob/main/test/xrt/09_gemm_extern_vec_4x4/aie.py) is listed below.

"buffer-results-to-out-params"  
["air-linalg-to-func{link-with=mm.o}"](#air-linalg-to-func)  
["air-par-to-herd{depth=1}"](#air-par-to-herd)  
["air-par-to-launch{has-air-segment=true}"](#air-par-to-launch)  
["air-copy-to-dma"](#air-copy-to-dma)   
"canonicalize", "cse"  
["air-dependency"](#air-dependency)  
["air-dependency-schedule-opt"](#air-dependency-schedule-opt)  
["air-specialize-dma-broadcast"](#air-specialize-dma-broadcast)  
["air-dma-to-channel"](#air-dma-to-channel)  
"canonicalize", "cse"  
["air-dependency-canonicalize"](#air-dependency-canonicalize)  
"canonicalize", "cse"  
['func.func(air-split-l2-memref)'](#air-split-l2-memref)  
["air-isolate-async-dma-loop-nests"](#air-isolate-async-dma-loop-nests)  
"canonicalize", "cse"  
["func.func(air-loop-fusion)"](#air-loop-fusion)  
["air-label-scf-for-to-ping-pong"](#air-label-scf-for-to-ping-pong)  
["air-ping-pong-transform{keep-memref-dealloc=true}"](#air-ping-pong-transform)  
"canonicalize", "cse"  
["air-specialize-channel-wrap-and-stride"](#air-specialize-channel-wrap-and-stride)  
"canonicalize", "cse"
["func.func(air-collapse-herd{max-col-size=4})"](#air-collapse-herd)  
'canonicalize', 'cse'  
["air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0}"](#air-place-herds)  
'canonicalize', 'cse'  
'func.func(air-renumber-dma)'
'canonicalize', 'cse'  
['air-to-aie{row-offset=2 col-offset=0 device=npu emit-while-loop=true}'](#air-to-aie)  
'canonicalize'
['air-to-std'](#air-to-std)  
'canonicalize'  
'symbol-dce'  
['func.func(affine-loop-opt{affine-opt-tile-sizes=4,4})'](#affine-loop-opt)  
['func.func(air-unroll-outer-affine-loops{depth=2})'](#air-unroll-outer-affine-loops)  
'affine-expand-index-ops'  
['airrt-to-npu'](#airrt-to-npu)  
'canonicalize'

## Overview

|Compilation Stage |Passes |Description |
|:--- |:--- |:--- |
|Convert to MLIR-AIR    |   <br> <ul><li>`air-linalg-to-func{link-with=mm.o}`</li><li>`air-par-to-herd{depth=1}`</li><li>`air-par-to-launch{has-air-segment=true}`</li><li>`air-copy-to-dma`</li></ul>    |   Binding parallelizable loops to `air` hierarchies; binding data movement operations to `air.dma_memcpy_nd` operations; binding linear algebra compute operations with link to AIE core kernel. |
|Asynchronous dependency analysis    |   <br> <ul><li>`air-dependency`</li><li>`air-dependency-canonicalize`</li></ul>    |   Construction of asynchronous task graph, as an explicit representation of the asynchronous concurrency in the hardware schedule. |
|Broadcast    |   <br> <ul><li>`air-dependency-schedule-opt`</li><li>`air-specialize-dma-broadcast`</li></ul>    |   Detection and lowering of broadcasting data movement to map to circuit-routed streaming interconnects. |
|Generate half-dma operations    |   <br> <ul><li>`air-dma-to-channel`</li></ul>    |   Lowering synchronous or asynchronous `air.dma_memcpy_nd` operations to `air.channel.put` or `air.channel.get` operations representing half-dma data sends and receives. |
|Outline L2 memrefs to memtile buffers    |   <br> <ul><li>`func.func(air-split-l2-memref)`</li></ul>    |   Tiling L2 memrefs based on parallelizable data movements, explicitly represented via `scf.parallel` or `air.channel.put/get` operations, in order to maximize memtile bandwidth utilization. |
|Memtile DMA BD Optimization    |   <br> <ul><li>`air-isolate-async-dma-loop-nests`</li><li>`func.func(air-loop-fusion)`</li><li>`air-specialize-channel-wrap-and-stride`</li></ul>    |   Lowering L2 control flow program into finite-state machines made of Block Descriptors as states. |
|Double buffering    |   <br> <ul><li>`air-label-scf-for-to-ping-pong`</li><li>`air-ping-pong-transform{keep-memref-dealloc=true}`</li></ul>    |   Detecting and lowering double buffering opportunities by analyzing data production and consumption patterns to a `memref` within an `scf.for` loop; explicitly represent the multiple asynchronous threads traversing through the loop. |
|Outline air.herd to aie.tiles    |   <br> <ul><li>`func.func(air-collapse-herd{max-col-size=4})`</li><li>`air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0}`</li><li>`func.func(air-renumber-dma)`</li></ul>    |   Reshaping and placing `air.herd` onto `air.segment`; inferring `air.segment` shape and size. |
|Convert MLIR-AIR to MLIR-AIE    |   <br> <ul><li>`func.func(air-renumber-dma)`</li><li>`air-to-aie{row-offset=2 col-offset=0 device=npu emit-while-loop=true}`</li></ul>    |   Converting to MLIR-AIE dialect. Clone the `func.func` op, where one copy lowers to the circuit design to be mapped onto AIE tiles, and the other copy lowers to LX6 control program; outline `air.herd` body into `aie.core` kernel; materialize asynchronous `air.channel.put/get` into dma block descriptors and `aie.lock`. |
|SHIM DMA BD Optimization    |   <br> <ul><li>`air-to-std`</li><li>`func.func(affine-loop-opt{affine-opt-tile-sizes=4,4})`</li><li>`func.func(air-unroll-outer-affine-loops{depth=2})`</li><li>`airrt-to-npu`</li></ul>    |   Converting the control code via AIRRt and AIEX.NPU dialect to NPU SHIM DMA instruction sequence. |
||||||

## MLIR-AIR Passes

### air-linalg-to-func

Links `linalg.generic` operation and variants with object file (.o) compiled from each `aie.core`'s compute kernel.

*Input IR:*
```
linalg.generic {library_call = "zero_f32", indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : f32) outs(%arg2 : memref<64x64xf32>) {
^bb0(%in: f32, %out: f32):
  linalg.yield %in : f32
}
linalg.generic {library_call = "matmul_f32", indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<64x256xf32>, memref<256x64xf32>) outs(%arg2 : memref<64x64xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
  %0 = arith.mulf %in, %in_0 : f32
  %1 = arith.addf %out, %0 : f32
  linalg.yield %1 : f32
}
```
The input IR has `linalg.generic` attribute `library_call` that specifies the name of an external library call.

*Output IR:*
```
func.func private @zero_f32(f32, memref<64x64xf32>) attributes {link_with = "test.o", llvm.emit_c_interface}
func.func private @matmul_f32(memref<64x256xf32>, memref<256x64xf32>, memref<64x64xf32>) attributes {link_with = "test.o", llvm.emit_c_interface}
call @zero_f32(%cst, %arg2) : (f32, memref<64x64xf32>) -> ()
call @matmul_f32(%arg0, %arg1, %arg2) : (memref<64x256xf32>, memref<256x64xf32>, memref<64x64xf32>) -> ()
```
The output IR generates the `func.func` declaration to the function call, and replaces the `linalg.generic` with `call`.
        
### air-par-to-herd

Converts parallel computations, represented by `scf.parallel` or `scf.forall`, into a more optimized, hardware-specific form called `air.herd`. This transformation is targeted towards accelerating the matrix multiplication problem, by partitioning the proglem into strictly spatial concurrent threads represented by the iteration space of `air.herd`.

*Input IR:*
```
module {
  func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
    ...
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
      ...
      scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c128, %c128) step (%c32, %c32) {
        ...
        scf.reduce 
      }
      ...
      scf.reduce 
    }
    return
  }
}
```
The input IR sets up a matrix multiplication operation, with input and output data references to memory via `memref` values. It includes MLIR operations which specify sub-views to memory and subdivide the work into smaller chunks that can be processed in parallel or in a loop, for performance optimization.

It features nested `scf.parallel` and `scf.for` loops to iterate over matrix elements, creating subviews and temporary buffers (`memref.alloc`) for storing intermediate results in each AIE tile's local (L1) memory. Inside these loops, `memref.copy` and `linalg.copy` operations are used to move data between different parts of memory, indicating a complex memory management pattern aimed at optimizing data locality and access patterns for the matrix multiplication.

*Output IR:*
```
module {
  func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
    ...
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
      ...
      air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c4_8, %arg5=%c4_9) args(%arg6=%alloc, %arg7=%alloc_0, %arg8=%alloc_1) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> {
        ...
        air.herd_terminator
      }
      ...
      scf.reduce 
    }
    return
  }
}
```
The transformed code replaces certain `scf.parallel` loop structures in the loop nest---specified by `depth` option---with the air.herd operation, where the `air.herd` encapsulates the parallelism in memory management and computation (like loading, computing, and storing results) that is common in matrix multiplications.

By consolidating the `scf.parallel` body with the `isolateFromAbove` `air.herd` interface, the `air.herd` operation reduces the overhead associated with loop management and memory operations within the parent region.

The L1 memory allocation, deallocation, and copying operations that prepare data for the `air.herd` operation are retained within its body. This ensures that the memory management within each of its parallel thread is still handled explicitly, to maintain control over data layout and access patterns.

### air-par-to-launch

Converts parallel computations, represented by `scf.parallel` or `scf.forall`, into `air.launch` construct. This transformation is targeted towards the dispatching of the already parallelized computations---represented by `air.herd`---in a more structured and *potentially* parallelized manner that is better suited for the dynamic launching of program iterations to reuse hardware configurations by the host.

*Input IR:*
```
module {
  func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
    ...
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c512, %c512) step (%c128, %c128) {
      ...
      air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c4_8, %arg5=%c4_9) args(%arg6=%alloc, %arg7=%alloc_0, %arg8=%alloc_1) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> {
        ...
        air.herd_terminator
      }
      ...
      scf.reduce 
    }
    return
  }
}
```
The input IR uses nested loops (`scf.parallel` and `scf.for`) to iterate over the chunks of the matrices and perform computations in a tiled manner, with `scf.parallel` representing parallelizable chunks and `scf.for` representing strictly sqeuential chunks due to loop-carried dependency.

Inside the parallel and for-loop constructs, there are detailed memory management operations (`memref.alloc`, `memref.copy`) and computational kernels (encapsulated in the `air.herd` operation).

*Output IR:*
```
module {
  func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
    ...
    air.launch (%arg0, %arg1) in (%arg2=%c4_6, %arg3=%c4_7) args(%arg4=%2, %arg5=%0, %arg6=%1) : memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> {
      air.segment @segment_0  args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2, %arg10=%arg3, %arg11=%arg4, %arg12=%arg5, %arg13=%arg6) : index, index, index, index, memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> {
        ...
        air.herd @herd_0  tile (%arg14, %arg15) in (%arg16=%c4_19, %arg17=%c4_20) args(%arg18=%alloc, %arg19=%alloc_11, %arg20=%alloc_12) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> {
          ...
          air.herd_terminator
        }
        ...
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
```
The transformed code introduces `air.launch` as the top-level construct, indicating the start of a parallelizable hardware-accelerated computation. Within `air.launch`, an `air.segment` construct is also optionally generated---controlled by option `has-air-segment`---representing a partition of the program, consisting of single or multiple `air.herd` and L2 memory operations, to be strictly spatially mapped to a spatially contiguous plot of AIE tiles and memtiles. The guaranteed spatial coexistence of all `air.herd` and L2 memory references ensure data flow between specialized AIE tiles via L2 memory.

The `air.segment` encapsulates the original computational logic, including L2 memory operations and the `air.herd` operation, which itself represents a computational kernel optimized for a matrix of AIE compute tiles. This encapsulation suggests that the computation can be offloaded in a virtualized AIE device cloud, as long as the resource requirements are fulfilled.

The transformation retains the explicit management of L2 memory (allocations and copies) and computations (the `air.herd` operation) but organizes them within the `air.launch` construct. This organization could facilitate the dynamic hardware dispatch managed from the host-side control program, scheduled by the compiler or runtime system, taking advantage of hardware capabilities for parallel execution.

### air-copy-to-dma

Converts memory operations to optimize data transfer through Direct Memory Access (DMA) operations, targeting AIE's DMA hardware.

*Input IR:*
```
scf.for %arg12 = %c0_1 to %c1024 step %c256 {
  %subview_4 = memref.subview %arg10[%3, %arg12] [128, 256] [1,1] : memref<512x1024xi32> to memref<128x256xi32, strided<[1024,1], offset: ?>>
  %subview_5 = memref.subview %alloc[0, %arg12] [128, 256] [1, 1]: memref<128x1024xi32, 1 : i32> to memref<128x256xi32, strided[1024, 1], offset: ?>, 1 : i32>
  memref.copy %subview_4, %subview_5 : memref<128x256xi32,strided<[1024, 1], offset: ?>> to memref<128x256xi32, strided[1024, 1], offset: ?>, 1 : i32>
}
```
The input IR contains some generic memory copy operations (`memref.copy` or `linalg.copy`), moving data between L1, L2 or L3 memories.


*Output IR:*
```
scf.for %arg12 = %c0_1 to %c1024 step %c256 {
  %subview_5 = memref.subview %arg10[%3, %arg12] [128, 256] [11] : memref<512x1024xi32> to memref<128x256xi32, stride[1024, 1], offset: ?>>
  %subview_6 = memref.subview %alloc[0, %arg12] [128, 256] [11] : memref<128x1024xi32, 1 : i32> to memref<128x256xi32strided<[1024, 1], offset: ?>, 1 : i32>
  air.dma_memcpy_nd (%alloc[%c0_11, %arg12] [%c128_12, %c256_13] [%c1024_14, %c1_15], %arg10[%3, %arg12] [%c128_7, %c256_8] [%c1024_9, %c1_10]) {id = 1 : i32} (memref<128x1024xi32, 1 : i32>, memref<512x1024xi32>)
}
```
The transformed code replaces some of the generic memory copy operations with `air.dma_memcpy_nd` operations. These DMA operations are specialized for direct memory-to-memory transfers, bypassing the host and thus reducing data transfer latency and host workload. The `air.dma_memcpy_nd` operation is designed to handle n-dimensional data transfers efficiently, explicitly representing the n-dimensional data rearrangement pattern on the fly.

Optimization of Data Transfer Dimensions: The transformation includes specifying the *offsets,* *sizes* and *strides* for the DMA operations explicitly, counted in number of elements (e.g. `[%c0_11, %arg12] [%c128_12, %c256_13] [%c1024_14, %c1_15]`).

Corresponding to the DMA operations, there are allocations (`memref.alloc()`) for temporary storage and deallocations (`memref.dealloc()`) for these temporary spaces after the DMA operations are complete. This ensures that the memory resources are managed efficiently and are only used when needed.

The core computational logic, represented by operations like `linalg.generic`, remains intact. The focus of the `air-copy-to-dma` pass is on optimizing the data movement that surrounds or supports these computations, rather than altering the computations themselves.

The DMA operations include identifiers (`{id = 1 : i32}, {id = 2 : i32}, {id = 3 : i32}`), which could be used for tracking, debugging, or further optimization by subsequent passes or by the runtime system. This explicit tagging could help in analyzing the performance or behavior of DMA operations within the execution pipeline.

### air-dependency

Transforms the original IR towards an asynchronous execution model, specifically targeting optimizations and parallel execution of memory operations, computational tasks, and data transfers.

*Input IR:*
A generic MLIR IR representing some compute or data movement operations to be executed on hardware.

*Output IR:*
The pass introduces asynchronous air.execute operations that encapsulate both data transfer (`air.dma_memcpy_nd`) and computational (`linalg.generic`) tasks. These tasks are now explicitly managed through asynchronous tokens (`%async_token`), which represent dependencies between operations. The `air.wait_all` operation is used to synchronize on multiple asynchronous tokens, effectively waiting for all dependent operations to complete before proceeding. This allows for overlapping data transfers with computation, reducing idle time and improving overall execution efficiency.

By leveraging asynchronous tokens and explicit waits (`air.wait_all`), the pass enforces a clear execution order based on data dependencies. This explicit management helps in ensuring that operations are executed as soon as their prerequisites are met, potentially in parallel, leading to better resource utilization and shorter execution times.

If the input code was organized as nests and segments of `air.herd`, `air.segment` and `air.launch`, the transformed code shall also explicitly manage how they can be executed asynchronously (`air.launch async`). This segmentation, along with the assignment of unique identifiers to operations ({id = X : i32}), suggests a fine-grained control over the execution flow, enabling optimizations like concurrent execution of independent code segments.

The pass separates data preparation (e.g., memory allocations and data transfers) from the computational workload. By doing so, it allows for data transfers to be overlapped with computations, minimizing the overall execution latency. This is particularly beneficial to AIEs featuring discrete hardware DMA, which is faster and more efficient than host-driven memory transfers, as they can move data directly between memory locations without host intervention.

Please check out this [link](https://github.com/Xilinx/mlir-air/blob/main/docs/AIRAsyncConcurrency.md) for a detailed description of this pass.

### air-dependency-schedule-opt

Optimizes the scheduling of dependencies in the MLIR module, specifically focusing on improving the parallelism and efficiency of execution through the asynchronous execution model. One key optimization introduced is the detection of broadcasting data movement opportunities, and the subsequent inferring of broadcasting pattern.

*Input IR:*
```
%6 = air.herd @herd_0 async [%async_token_13, %async_token_15, %async_token_17]  tile (%arg12, %arg13) in (%arg14=%c4_7, %arg15=%c4_7) args(%arg16=%results_14, %arg17=%results_16, %arg18=%results_18) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> attributes {id = 1 : i32} {
  ..
  %async_token_31, %results_32 = air.execute -> (memref<8x8x4x4xi32, 2   32>) {
    %alloc = memref.alloc() : memref<8x8x4x4xi32, 2 : i32>
    air.execute_terminator %alloc : memref<8x8x4x4xi32, 2 : i32>
  } {id = 14 : i32}
  %async_token_33 = air.execute [%async_token_31] {
    linalg.fill ins(%c0_i32 : i32) outs(%results_32 : memref<8x8x4x4xi32, 2 : i32>)
  } {id = 15 : i32}
  %8 = air.wait_all async [%async_token_27, %async_token_29, %async_token_33]  {id = 12 : i32}
  %9 = scf.for %arg19 = %c0_23 to %c128_26 step %c4_24 iter_args(%arg20 = %8) -> (!air.async.token) {
    ...
    %11 = air.dma_memcpy_nd async [%async_token_44, %async_token_42, %arg20] (%results_45[%c0_35] [%c1024_36] [%c1_37], %arg16[%c0_35, %results_28, %results_43] [%c4_38, %c32_39, %c8_40] [%c8_40, %c1024_36, %c1_37]) {id = 3 : i32} : (memref<4x8x4x8xi32, 2 : i32>, memref<128x1024xi32, 1 : i32>)
    %async_token_46, %results_47 = air.execute -> (memref<8x4x8x4xi32, 2 : i32>) {
      %alloc = memref.alloc() : memref<8x4x8x4xi32, 2 : i32>
      air.execute_terminator %alloc : memref<8x4x8x4xi32, 2 : i32>
    } {id = 18 : i32}
    %12 = air.dma_memcpy_nd async [%async_token_46, %async_token_42, %arg20] (%results_47[%c0_35] [%c1024_36] [%c1_37], %arg17[%c0_35, %results_43, %results_30] [%c8_40, %c32_39, %c4_38] [%c4_38, %c128_41, %c1_37]) {id = 4 : i32} : (memref<8x4x8x4xi32, 2 : i32>, memref<1024x128xi32, 1 : i32>)
    ...
    scf.yield %15 : !air.async.token
  }
  %10 = air.dma_memcpy_nd async [%async_token_27, %async_token_29, %async_token_33] (%arg18[%results_28, %results_30] [%c32, %c32] [%c128_26, %c1_25], %results_32[%c0_23, %c0_23, %c0_23] [%c32, %c8, %c4_24] [%c4_24, %c128_26, %c1_25]) {id = 5 : i32} : (memref<128x128xi32, 1 : i32>, memref<8x8x4x4xi32, 2 : i32>)
  ...
  air.herd_terminator
}
```
The input IR has the L1 memory management and computation encapsulated within `air.herd`, and any direct memory-to-memory transfers represented with `air.dma_memcpy_nd` operations.

*Output IR:*
```
%6 = air.herd @herd_0 async [%async_token_13, %async_token_15, %async_token_17]  tile (%arg12, %arg13) in (%arg14=%c4_7, %arg15=%c4_7) args(%arg16=%results_14, %arg17=%results_16, %arg18=%results_18) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> attributes {id = 1 : i32} {
  ...
  %9 = scf.for %arg19 = %c0_23 to %c128_26 step %c4_24 iter_args(%arg20 = %8) -> (!air.async.token) {
    ...
    %async_token_37, %results_38 = air.execute -> (memref<4x8x4x8xi32, 2 : i32>) {
      %alloc = memref.alloc() : memref<4x8x4x8xi32, 2 : i32>
      air.execute_terminator %alloc : memref<4x8x4x8xi32, 2 : i32>
    } {id = 17 : i32}
    %11 = air.dma_memcpy_nd async [%async_token_37, %async_token_35, %arg20] (%results_38[%c0_23] [%c1024_22] [%c1_25], %arg16[%c0_23, %results_28, %results_36] [%c4_24, %c32, %c8] [%c8, %c1024_22, %c1_25]) {broadcast_pattern = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 3 >= 0, s0 >= 0, -s0 + 3 >= 0)>, id = 3 : i32} : (memref<4x8x4x8xi32, 2 : i32>, memref<128x1024xi32, 1 : i32>)
    %async_token_39, %results_40 = air.execute -> (memref<8x4x8x4xi32, 2 : i32>) {
      %alloc = memref.alloc() : memref<8x4x8x4xi32, 2 : i32>
      air.execute_terminator %alloc : memref<8x4x8x4xi32, 2 : i32>
    } {id = 18 : i32}
    %12 = air.dma_memcpy_nd async [%async_token_39, %async_token_35, %arg20] (%results_40[%c0_23] [%c1024_22] [%c1_25], %arg17[%c0_23, %results_36, %results_30] [%c8, %c32, %c4_24] [%c4_24, %c128_26, %c1_25]) {broadcast_pattern = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 3 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 3 >= 0)>, id = 4 : i32} : (memref<8x4x8x4xi32, 2 : i32>, memref<1024x128xi32, 1 : i32>)
    ...
    scf.yield %15 : !air.async.token
  }
  %10 = air.dma_memcpy_nd async [%async_token_27, %async_token_29, %async_token_33] (%arg18[%results_28, %results_30] [%c32, %c32] [%c128_26, %c1_25], %results_32[%c0_23, %c0_23, %c0_23] [%c32, %c8, %c4_24] [%c4_24, %c128_26, %c1_25]) {id = 5 : i32} : (memref<128x128xi32, 1 : i32>, memref<8x8x4x4xi32, 2 : i32>)
  ...
  air.herd_terminator
}
```
The transformation pass optimizes some `air.dma_memcpy_nd` operations within the body of `air.herd` by introducing a `broadcast_pattern` attribute. The presence of a `broadcast_pattern` attribute in an `air.dma_memcpy_nd` operation suggests that the data transfer is being optimized to replicate data across multiple destinations in a specific pattern. 

`air.herd` orchestrates parallel execution of computational tasks across multiple AIE tiles, and
specifying a broadcast_pattern for data transfers in its body reduces the bandwidth bottlenecks related to the finite AIE DMA channels.

The specific details of the broadcast_pattern attribute (e.g., `affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 3 >= 0, s0 >= 0, -s0 + 3 >= 0)>`) describe the rules that govern the data  replication pattern. Specifically, the symbol `s0` represents an iteration across unique broadcast sources, while the dimensions `d0` and `d1` represent the two-dimensional broadcast destination space: the set in `d0, d1` space corresponding to a `true` result in the `affine_set` represents all broadcast destinations to the broacast source `s0`. For example, in the `affine_set` given above, for each integer value of `s0`, ranged within `[0, 3]`, the set returns `true` if `(d0 == s0, 0 <= d1 <= 3`, meaning that this DMA operation shall involve four unique broadcast sources, where each source is broadcasted four-way across the 2nd (`d1`) dimension.

### air-specialize-dma-broadcast

Specializes `air.dma_memcpy_nd` operations for broadcast patterns within a computation. This specialization involves transforming data movement operations into more optimized versions that are aware of the broadcast semantics.

*Input IR*
```
...
%6 = air.herd @herd_0 async [%async_token_13, %async_token_15, %async_token_17]  tile (%arg12, %arg13) in (%arg14=%c4_7, %arg15=%c4_7) args(%arg16=%results_14, %arg17=%results_16, %arg18=%results_18) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> attributes {id = 1 : i32} {
  ...
  %11 = air.dma_memcpy_nd async [%async_token_37, %async_token_35, %arg20] (%results_38[%c0_23] [%c1024_22] [%c1_25], %arg16[%c0_23, %results_28, %results_36] [%c4_24, %c32, %c8] [%c8, %c1024_22, %c1_25]) {broadcast_pattern = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 3 >= 0, s0 >= 0, -s0 + 3 >= 0)>, id = 3 : i32} : (memref<4x8x4x8xi32, 2 : i32>, memref<128x1024xi32, 1 : i32>)
  ...
  air.herd_terminator
}
...
```
The input IR has the L1 memory management and computation encapsulated within `air.herd`, and any direct memory-to-memory transfers represented with `air.dma_memcpy_nd` operations, some of which annotated with a `broadcast_pattern` attribute, indicating broadcasting data movement opportunity.

*Output IR*
```
...
%6 = air.herd @herd_0 async [%async_token_13, %async_token_15, %async_token_17]  tile (%arg12, %arg13) in (%arg14=%c4_7, %arg15=%c4_7) args(%arg16=%results_14, %arg17=%results_16, %arg18=%results_18) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> attributes {id = 1 : i32} {
  ...
  %11 = affine.if affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>()[%arg12, %arg13] -> !air.async.token {
    ...
    %16 = air.dma_memcpy_nd async [%async_token_37, %async_token_35, %arg20] (%results_38[%c0_23] [%c1024_22] [%c1_25], %arg16[%c0_44, %c0_43, %results_36] [%c4_24, %c32, %c8] [%c8, %c1024_22, %c1_25]) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>, id = 3 : i32} : (memref<4x8x4x8xi32, 2 : i32>, memref<128x1024xi32, 1 : i32>)
    affine.yield %16 : !air.async.token
  } else {
    %16 = affine.if affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>()[%arg12, %arg13] -> !air.async.token {
      ...
      %17 = air.dma_memcpy_nd async [%async_token_37, %async_token_35, %arg20] (%results_38[%c0_23] [%c1024_22] [%c1_25], %arg16[%c0_44, %c32_43, %results_36] [%c4_24, %c32, %c8] [%c8, %c1024_22, %c1_25]) {broadcast_set = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>, id = 4 : i32} : (memref<4x8x4x8xi32, 2 : i32>, memref<128x1024xi32, 1 : i32>)
      affine.yield %17 : !air.async.token
    } else {
      %17 = affine.if affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>()[%arg12, %arg13] -> !air.async.token {
        ...
        %18 = air.dma_memcpy_nd async [%async_token_37, %async_token_35, %arg20] (%results_38[%c0_23] [%c1024_22] [%c1_25], %arg16[%c0_43, %c64, %results_36] [%c4_24, %c32, %c8] [%c8, %c1024_22, %c1_25]) {broadcast_set = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>, id = 5 : i32} : (memref<4x8x4x8xi32, 2 : i32>, memref<128x1024xi32, 1 : i32>)
        affine.yield %18 : !air.async.token
      } else {
        ...
        %18 = air.dma_memcpy_nd async [%async_token_37, %async_token_35, %arg20] (%results_38[%c0_23] [%c1024_22] [%c1_25], %arg16[%c0_43, %c96, %results_36] [%c4_24, %c32, %c8] [%c8, %c1024_22, %c1_25]) {broadcast_set = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>, id = 6 : i32} : (memref<4x8x4x8xi32, 2 : i32>, memref<128x1024xi32, 1 : i32>)
        affine.yield %18 : !air.async.token
      }
      affine.yield %17 : !air.async.token
    }
    affine.yield %16 : !air.async.token
  }
  ...
  air.herd_terminator
}
...
```
The output IR uses `affine.if` operations alongside the `air.dma_memcpy_nd` operations. This conditional execution is based on evaluating specific broadcast conditions, represented by affine sets, that determine which AIE tiles are subject to data broadcasting, and how data should be broadcasted.

The broadcast set annotations specify the conditions under which a particular broadcasting behavior should be applied, which is derived from the analysis of the `broadcast_pattern` attribute generated by a prior pass (`air-dependency-schedule-opt`). 

By specializing the `air.dma_memcpy_nd` operations for broadcast patterns, this pass reduces the onchip bandwidth usage associated with generic data movement. For instance, if certain data needs to be broadcasted across multiple destination AIE tiles in a specific pattern, the specialized code can reduce the number of DMA channels on the source tile, while maintaining the same data movement speed.

### air-dma-to-channel

Transforms direct memory access (DMA) operations into channel-based communications, consisting of a series of channel put and get operations via shared channel constructs.

*Input IR:*
```
%0 = air.launch async [%async_token_0, %async_token_3, %async_token_6] (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) args(%arg4=%results_5, %arg5=%results, %arg6=%results_2) : memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 3 : i32} {
  %1 = air.segment @segment_0 async  args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg4, %arg10=%arg5, %arg11=%arg6) : index, index, memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 2 : i32} {
    ...
    %3 = scf.for %arg12 = %c0_8 to %c1024 step %c256 iter_args(%arg13 = %2) -> (!air.async.token) {
      %8 = air.dma_memcpy_nd async [%arg13, %arg13] (%results_14[%c0_8, %arg12] [%c128, %c256] [%c1024, %c1], %arg10[%results_10, %arg12] [%c128, %c256] [%c1024, %c1]) {id = 1 : i32} : (memref<128x1024xi32, 1 : i32>, memref<512x1024xi32>)
      ...
    }
    %6 = air.herd @herd_0 async [%async_token_13, %async_token_15, %async_token_17]  tile (%arg12, %arg13) in (%arg14=%c4_7, %arg15=%c4_7) args(%arg16=%results_14, %arg17=%results_16, %arg18=%results_18) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> attributes {id = 1 : i32} {
      ...
      %9 = scf.for %arg19 = %c0_23 to %c128_26 step %c4_24 iter_args(%arg20 = %8) -> (!air.async.token) {
        ...
        %16 = air.dma_memcpy_nd async [%async_token_37, %async_token_35, %arg20] (%results_38[%c0_23] [%c1024_22] [%c1_25], %arg16[%c0_44, %c0_43, %results_36] [%c4_24, %c32, %c8] [%c8, %c1024_22, %c1_25]) {broadcast_set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>, id = 3 : i32} : (memref<4x8x4x8xi32, 2 : i32>, memref<128x1024xi32, 1 : i32>)
        ...
      }
      ...
      air.herd_terminator
    }
    ...
    air.segment_terminator
  }
  air.launch_terminator
}
```
The input IR has the L1 memory management and computation encapsulated within `air.herd`, the L2 memory management optionally encapsulate within `air.segment`, and any direct memory-to-memory transfers represented with `air.dma_memcpy_nd` operations. The input IR may or may not have MLIR-AIR asynchronous interface.

*Output IR:*
```
...
air.channel @channel_8 [1, 1]
...
air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
...
%0 = air.launch async [%async_token_0, %async_token_3, %async_token_6] (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) args(%arg4=%results_5, %arg5=%results, %arg6=%results_2) : memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 3 : i32} {
  ...
  %2 = scf.for %arg7 = %c0_7 to %c1024 step %c256 iter_args(%arg8 = %1) -> (!air.async.token) {
    ...
    %17 = air.channel.put async [%async_token_8, %arg8]  @channel_8[] (%arg5[%results_9, %arg7] [%c128, %c256] [%c1024, %c1]) : (memref<512x1024xi32>)
    ...
  }
  ...
  %16 = air.segment @segment_0 async  args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg4, %arg10=%arg5, %arg11=%arg6) : index, index, memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 2 : i32} {
    ...
    %18 = scf.for %arg12 = %c0_32 to %c1024_33 step %c256_34 iter_args(%arg13 = %17) -> (!air.async.token) {
      %49 = air.channel.get async [%arg13, %arg13]  @channel_8[] (%results_40[%c0_32, %arg12] [%c128_30, %c256_34] [%c1024_33, %c1_29]) : (memref<128x1024xi32, 1 : i32>)
      ...
    }
    ...
    %23 = scf.for %arg12 = %c0_47 to %c128_50 step %c4_48 iter_args(%arg13 = %22) -> (!air.async.token) {
      ...
      %49 = air.channel.put async [%async_token_160, %async_token_39, %arg13]  @channel_0[] (%results_40[%c0_163, %c0_162, %results_161] [%c4_48, %c32, %c8] [%c8, %c1024_46, %c1_49]) : (memref<128x1024xi32, 1 : i32>)
      ...
    }
    ...
    %47 = air.herd @herd_0 async [%async_token_39, %async_token_41, %async_token_43]  tile (%arg12, %arg13) in (%arg14=%c4_31, %arg15=%c4_31) args(%arg16=%results_40, %arg17=%results_42, %arg18=%results_44) : memref<128x1024xi32, 1 : i32>, memref<1024x128xi32, 1 : i32>, memref<128x128xi32, 1 : i32> attributes {id = 1 : i32} {
      ...
      %50 = scf.for %arg19 = %c0_155 to %c128_159 step %c4_156 iter_args(%arg20 = %49) -> (!air.async.token) {
        ...
        %57 = air.channel.get async [%async_token_170, %async_token_168, %arg20]  @channel_0[%arg12, %arg13] (%results_171[%c0_155] [%c1024_154] [%c1_158]) : (memref<4x8x4x8xi32, 2 : i32>)
        ...
      }
      ...
      air.herd_terminator
    }
    air.segment_terminator
  }
  air.launch_terminator
}
```
The transformation introduces various `air.channel` entities, each configured for specific communication patterns (e.g., sizes `[1, 1]` or `[4, 4]`, `broadcast_shapes`). Direct memory accesses (`air.dma_memcpy_nd`) are replaced by channel operations (`air.channel.put`, `air.channel.get`), explicitly representing data being sent or received from memory through channels, i.e. AIE tile-to-tile streaming interconnect.

The use of channels for communication also implies a synchronization mechanism. When data is put into or retrieved from a channel, the operations are inherently synchronized with the data's availability, introducing more deterministic execution patterns compared to direct memory access.

By organizing data movement through channels, the transformed IR is better suited for parallel execution across AIE tiles, where channels can facilitate the broadcasting of data across multiple AIE tiles.

### air-dependency-canonicalize

Transforms the input IR by optimizing and restructuring the dependency and execution flow of the operations without altering the semantics of the original program. 

*Input IR:*
```
...
%8 = scf.for %arg9 = %c0_23 to %c1024_24 step %c256_25 iter_args(%arg10 = %7) -> (!air.async.token) {
  %22 = air.channel.get async [%arg10, %arg10]  @channel_8[] (%results_31[%c0_23, %arg9] [%c128_21, %c256_25] [%c1024_24, %c1_20]) : (memref<128x1024xi32, 1 : i32>)
  %23 = air.wait_all async [%arg10, %22]  {id = 1 : i32}
  scf.yield %23 : !air.async.token
}
...
%19 = scf.parallel (%arg9, %arg10) = (%c0_23, %c0_23) to (%c4_22, %c4_22) step (%c1_20, %c1_20) init (%async_token_34) -> !air.async.token {
  %async_token_55, %results_56 = air.execute -> (index) {
    %23 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg9]
    air.execute_terminator %23 : index
  } {id = 12 : i32}
  %async_token_57, %results_58 = air.execute -> (index) {
    %23 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg10]
    air.execute_terminator %23 : index
  } {id = 13 : i32}
  %22 = air.channel.get async [%async_token_57, %async_token_55, %async_token_34, %async_token_34]  @channel_10[%arg9, %arg10] (%results_35[%results_56, %results_58] [%c32, %c32] [%c128_21, %c1_20]) : (memref<128x128xi32, 1 : i32>)
  scf.reduce(%22 : !air.async.token) {
  ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
    %23 = air.wait_all async [%arg11, %arg12] 
    scf.reduce.return %23 : !air.async.token
  }
}
...
```
The pass consists of asynchronous data transfer, computational and synchronization tasks, explicitly managed through asynchronous tokens (`%async_token`), which represent dependencies between operations.

*Output IR:*
```
...
%7 = scf.for %arg9 = %c0_19 to %c1024_20 step %c256_21 iter_args(%arg10 = %async_token_22) -> (!air.async.token) {
  %20 = air.channel.get async [%arg10]  @channel_8[] (%results_23[%c0_19, %arg9] [%c128_17, %c256_21] [%c1024_20, %c1_16]) {id = 4 : i32} : (memref<128x1024xi32, 1 : i32>)
  scf.yield %20 : !air.async.token
}
...
%17 = scf.parallel (%arg9, %arg10) = (%c0_19, %c0_19) to (%c4_18, %c4_18) step (%c1_16, %c1_16) init (%async_token_26) -> !air.async.token {
  %async_token_31, %results_32 = air.execute -> (index) {
    %21 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg9]
    air.execute_terminator %21 : index
  }
  %async_token_33, %results_34 = air.execute -> (index) {
    %21 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg10]
    air.execute_terminator %21 : index
  }
  %20 = air.channel.get async [%async_token_26, %async_token_33, %async_token_31]  @channel_10[%arg9, %arg10] (%results_27[%results_32, %results_34] [%c32, %c32] [%c128_17, %c1_16]) {id = 14 : i32} : (memref<128x128xi32, 1 : i32>)
  scf.reduce(%20 : !air.async.token) {
  ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
    %21 = air.wait_all async [%arg11, %arg12] 
    scf.reduce.return %21 : !air.async.token
  }
}
...
```
The transformation pass reduces the unnecessary propagation and handling of async tokens in the task graph, leading to cleaner and potentially more efficient code representation.

The pass also optimizes unnecessary asynchronous events, including memory allocation, deallocation and scalar arithmetic operations (`memref.alloc,` `memref.dealloc,` `arith` and `affine.appy`), by analyzing the lifetime and usage of their results and yielded async tokens, thus reducing overhead and improving memory usage efficiency.

The pass simplifies the loop-carred async tokens in control flow constructs (`scf.for`, `scf.parallel`) and potentially merges or eliminates redundant control flow paths. This makes the program easier to understand and can help in further optimization passes.

### air-split-l2-memref

Transforms the input IR by splitting certain L2 memory references (`memrefs`) to adhere to AIE memtile-specific buffer and DMA channel constraints or optimization opportunities.

*Input IR:*
```
%0 = air.launch async [%async_token_0, %async_token_3, %async_token_6] (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) args(%arg4=%results_5, %arg5=%results, %arg6=%results_2) : memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 1 : i32} {
  ...
  %1 = scf.for %arg7 = %c0_7 to %c1024 step %c256 iter_args(%arg8 = %async_token_8) -> (!air.async.token) {
    %5 = air.channel.put async [%arg8]  @channel_8[] (%arg5[%results_9, %arg7] [%c128, %c256] [%c1024, %c1]) {id = 1 : i32} : (memref<512x1024xi32>)
    scf.yield %5 : !air.async.token
  }
  ...
  %4 = air.segment @segment_0 async  attributes {id = 2 : i32} {
    ...
    %async_token_22, %results_23 = air.execute -> (memref<128x1024xi32, 1 : i32>) {
      %alloc = memref.alloc() : memref<128x1024xi32, 1 : i32>
      air.execute_terminator %alloc : memref<128x1024xi32, 1 : i32>
    }
    %7 = scf.for %arg7 = %c0_19 to %c1024_20 step %c256_21 iter_args(%arg8 = %async_token_22) -> (!air.async.token) {
      %20 = air.channel.get async [%arg8]  @channel_8[] (%results_23[%c0_19, %arg7] [%c128_17, %c256_21] [%c1024_20, %c1_16]) {id = 4 : i32} : (memref<128x1024xi32, 1 : i32>)
      scf.yield %20 : !air.async.token
    }
    ...
    %9 = scf.for %arg7 = %c0_19 to %c128_17 step %c4_18 iter_args(%arg8 = %async_token_22) -> (!air.async.token) {
      %async_token_31, %results_32 = air.execute [%arg8] -> (index) {
        %21 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
        air.execute_terminator %21 : index
      }
      %20 = air.channel.put async [%async_token_31]  @channel_0[] (%results_23[%c0_19, %c0_19, %results_32] [%c4_18, %c32, %c8] [%c8, %c1024_20, %c1_16]) {id = 6 : i32} : (memref<128x1024xi32, 1 : i32>)
      scf.yield %20 : !air.async.token
    }
    %10 = scf.for %arg7 = %c0_19 to %c128_17 step %c4_18 iter_args(%arg8 = %async_token_22) -> (!air.async.token) {
      %async_token_31, %results_32 = air.execute [%arg8] -> (index) {
        %21 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
        air.execute_terminator %21 : index
      }
      %20 = air.channel.put async [%async_token_31]  @channel_1[] (%results_23[%c0_19, %c32, %results_32] [%c4_18, %c32, %c8] [%c8, %c1024_20, %c1_16]) {id = 7 : i32} : (memref<128x1024xi32, 1 : i32>)
      scf.yield %20 : !air.async.token
    }
    %11 = scf.for %arg7 = %c0_19 to %c128_17 step %c4_18 iter_args(%arg8 = %async_token_22) -> (!air.async.token) {
      %async_token_31, %results_32 = air.execute [%arg8] -> (index) {
        %21 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
        air.execute_terminator %21 : index
      }
      %20 = air.channel.put async [%async_token_31]  @channel_2[] (%results_23[%c0_19, %c64, %results_32] [%c4_18, %c32, %c8] [%c8, %c1024_20, %c1_16]) {id = 8 : i32} : (memref<128x1024xi32, 1 : i32>)
      scf.yield %20 : !air.async.token
    }
    %12 = scf.for %arg7 = %c0_19 to %c128_17 step %c4_18 iter_args(%arg8 = %async_token_22) -> (!air.async.token) {
      %async_token_31, %results_32 = air.execute [%arg8] -> (index) {
        %21 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
        air.execute_terminator %21 : index
      }
      %20 = air.channel.put async [%async_token_31]  @channel_3[] (%results_23[%c0_19, %c96, %results_32] [%c4_18, %c32, %c8] [%c8, %c1024_20, %c1_16]) {id = 9 : i32} : (memref<128x1024xi32, 1 : i32>)
      scf.yield %20 : !air.async.token
    }
    ...
    air.segment_terminator
  }
  air.launch_terminator
}
```
The input IR consists of some L2 memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations.

*Output IR:*
```
%0 = air.launch async [%async_token_0, %async_token_3, %async_token_6] (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) args(%arg4=%results_5, %arg5=%results, %arg6=%results_2) : memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 1 : i32} {
  ...
  %5 = scf.for %arg7 = %c0_7 to %c1024 step %c256 iter_args(%arg8 = %async_token_8) -> (!air.async.token) {
    %21 = air.channel.put async [%arg8]  @channel_12[%c0_7, %c0_7] (%arg5[%1, %arg7] [%c32, %c256] [%c1024, %c1]) {id = 1 : i32} : (memref<512x1024xi32>)
    %22 = air.channel.put async [%arg8]  @channel_12[%c1, %c0_7] (%arg5[%2, %arg7] [%c32, %c256] [%c1024, %c1]) {id = 2 : i32} : (memref<512x1024xi32>)
    %23 = air.channel.put async [%arg8]  @channel_12[%c2, %c0_7] (%arg5[%3, %arg7] [%c32, %c256] [%c1024, %c1]) {id = 3 : i32} : (memref<512x1024xi32>)
    %24 = air.channel.put async [%arg8]  @channel_12[%c3, %c0_7] (%arg5[%4, %arg7] [%c32, %c256] [%c1024, %c1]) {id = 4 : i32} : (memref<512x1024xi32>)
    %25 = air.wait_all async [%21, %22, %23, %24] 
    scf.yield %25 : !air.async.token
  }
  ...
  %20 = air.segment @segment_0 async  attributes {id = 2 : i32} {
    ...
    %async_token_53, %results_54 = air.execute -> (memref<32x1024xi32, 1>) {
      %alloc = memref.alloc() : memref<32x1024xi32, 1>
      air.execute_terminator %alloc : memref<32x1024xi32, 1>
    }
    %async_token_55, %results_56 = air.execute -> (memref<32x1024xi32, 1>) {
      %alloc = memref.alloc() : memref<32x1024xi32, 1>
      air.execute_terminator %alloc : memref<32x1024xi32, 1>
    }
    %async_token_57, %results_58 = air.execute -> (memref<32x1024xi32, 1>) {
      %alloc = memref.alloc() : memref<32x1024xi32, 1>
      air.execute_terminator %alloc : memref<32x1024xi32, 1>
    }
    %async_token_59, %results_60 = air.execute -> (memref<32x1024xi32, 1>) {
      %alloc = memref.alloc() : memref<32x1024xi32, 1>
      air.execute_terminator %alloc : memref<32x1024xi32, 1>
    }
    ...
    %25 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %24) -> (!air.async.token) {
      %68 = air.channel.get async [%arg8]  @channel_12[%c0_22, %c0_22] (%results_54[%c0_50, %arg7] [%c32_18, %c256_24] [%c1024_51, %c1_52]) {id = 13 : i32} : (memref<32x1024xi32, 1>)
      %69 = air.channel.get async [%arg8]  @channel_12[%c1_19, %c0_22] (%results_56[%c0_43, %arg7] [%c32_18, %c256_24] [%c1024_44, %c1_45]) {id = 14 : i32} : (memref<32x1024xi32, 1>)
      %70 = air.channel.get async [%arg8]  @channel_12[%c2_17, %c0_22] (%results_58[%c0_36, %arg7] [%c32_18, %c256_24] [%c1024_37, %c1_38]) {id = 15 : i32} : (memref<32x1024xi32, 1>)
      %71 = air.channel.get async [%arg8]  @channel_12[%c3_16, %c0_22] (%results_60[%c0_29, %arg7] [%c32_18, %c256_24] [%c1024_30, %c1_31]) {id = 16 : i32} : (memref<32x1024xi32, 1>)
      %72 = air.wait_all async [%68, %69, %70, %71] 
      scf.yield %72 : !air.async.token
    }
    ...
    %31 = scf.for %arg7 = %c0_22 to %c128_20 step %c4_21 iter_args(%arg8 = %30) -> (!air.async.token) {
      %async_token_177, %results_178 = air.execute [%arg8] -> (index) {
        %69 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
        air.execute_terminator %69 : index
      }
      %68 = air.channel.put async [%async_token_177]  @channel_0[] (%results_54[%c0_22, %c0_46, %results_178] [%c4_21, %c32_18, %c8] [%c8_47, %c1024_48, %c1_49]) {id = 21 : i32} : (memref<32x1024xi32, 1>)
      scf.yield %68 : !air.async.token
    }
    %32 = air.wait_all async [%async_token_55] 
    %33 = scf.for %arg7 = %c0_22 to %c128_20 step %c4_21 iter_args(%arg8 = %32) -> (!air.async.token) {
      %async_token_177, %results_178 = air.execute [%arg8] -> (index) {
        %69 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
        air.execute_terminator %69 : index
      }
      %68 = air.channel.put async [%async_token_177]  @channel_1[] (%results_56[%c0_22, %c0_39, %results_178] [%c4_21, %c32_18, %c8] [%c8_40, %c1024_41, %c1_42]) {id = 22 : i32} : (memref<32x1024xi32, 1>)
      scf.yield %68 : !air.async.token
    }
    %34 = air.wait_all async [%async_token_57] 
    %35 = scf.for %arg7 = %c0_22 to %c128_20 step %c4_21 iter_args(%arg8 = %34) -> (!air.async.token) {
      %async_token_177, %results_178 = air.execute [%arg8] -> (index) {
        %69 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
        air.execute_terminator %69 : index
      }
      %68 = air.channel.put async [%async_token_177]  @channel_2[] (%results_58[%c0_22, %c0_32, %results_178] [%c4_21, %c32_18, %c8] [%c8_33, %c1024_34, %c1_35]) {id = 23 : i32} : (memref<32x1024xi32, 1>)
      scf.yield %68 : !air.async.token
    }
    %36 = air.wait_all async [%async_token_59] 
    %37 = scf.for %arg7 = %c0_22 to %c128_20 step %c4_21 iter_args(%arg8 = %36) -> (!air.async.token) {
      %async_token_177, %results_178 = air.execute [%arg8] -> (index) {
        %69 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
        air.execute_terminator %69 : index
      }
      %68 = air.channel.put async [%async_token_177]  @channel_3[] (%results_60[%c0_22, %c0_25, %results_178] [%c4_21, %c32_18, %c8] [%c8_26, %c1024_27, %c1_28]) {id = 24 : i32} : (memref<32x1024xi32, 1>)
      scf.yield %68 : !air.async.token
    }
    ...
    air.segment_terminator
  }
  air.launch_terminator
}

```
Large L2 `memrefs` are split into smaller chunks, if the `air.channel` data access pattern implies spatial parallelism. `air.channel` data access pattern may infer memref splitting opportunity if multiple data consumers or producers access chunks of the memory in non-overlapping pattern. This pattern is explicitly represented via `air.channel`, where the `memref` is accessed by multiple puts or gets via either multiple `air.channel` declarations, or multiple sub-channels of an `air.channel` declaration.

Once L2 memref splitting opportunities are detected, the pass then replaces them with smaller, more manageable pieces, and introduces new channel declarations (`@channel_X`) for each new memref, making downstream memtile buffer allocator easier to manage and optimize for bandwidth utilization.

The pass introduces explicit memory allocation and deallocation operations for the newly created L2 memrefs, thus explicitly managing their lifetime and facilitating their bufferization onto memtiles in the downstream pass.

### air-isolate-async-dma-loop-nests

Transforms the IR by splitting the `scf.for` loop nests in `air.segment` body, based on the loop-carried async dependency of the original loop. The goal is to attempt to split nested `scf.for` loops around `air.channel.put/get` operations into perfect `scf.for` loop nests where only a single `air.channel.put`, or `air.channel.get` operation exists. This exposes DMA channel optimization opportunities, where perfectly nested parent for loops can fold as extra DMA channel wrap-and-stride dimensions.

*Input IR:*
```
%0 = air.launch async [%async_token_0, %async_token_3, %async_token_6] (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) args(%arg4=%results_5, %arg5=%results, %arg6=%results_2) : memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 1 : i32} {
  ...
  %20 = air.segment @segment_0 async  attributes {id = 2 : i32} {
    ...
    %25 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %24) -> (!air.async.token) {
      %68 = air.channel.get async [%arg8]  @channel_12[%c0_22, %c0_22] (%results_54[%c0_50, %arg7] [%c32_18, %c256_24] [%c1024_51, %c1_52]) {id = 13 : i32} : (memref<32x1024xi32, 1>)
      %69 = air.channel.get async [%arg8]  @channel_12[%c1_19, %c0_22] (%results_56[%c0_43, %arg7] [%c32_18, %c256_24] [%c1024_44, %c1_45]) {id = 14 : i32} : (memref<32x1024xi32, 1>)
      %70 = air.channel.get async [%arg8]  @channel_12[%c2_17, %c0_22] (%results_58[%c0_36, %arg7] [%c32_18, %c256_24] [%c1024_37, %c1_38]) {id = 15 : i32} : (memref<32x1024xi32, 1>)
      %71 = air.channel.get async [%arg8]  @channel_12[%c3_16, %c0_22] (%results_60[%c0_29, %arg7] [%c32_18, %c256_24] [%c1024_30, %c1_31]) {id = 16 : i32} : (memref<32x1024xi32, 1>)
      %72 = air.wait_all async [%68, %69, %70, %71] 
      scf.yield %72 : !air.async.token
    }
    ...
    air.segment_terminator
  }
  air.launch_terminator
}
```
The input IR consists of some L2 memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations under some imperfect `scf.for` loop nest. L2 memory operations have been encapsulated within `air.segment`.

*Output IR:*
```
%0 = air.launch async [%async_token_0, %async_token_3, %async_token_6] (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) args(%arg4=%results_5, %arg5=%results, %arg6=%results_2) : memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 1 : i32} {
  ...
  %40 = air.segment @segment_0 async  attributes {id = 2 : i32} {
    ...
    %47 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %46) -> (!air.async.token) {
      %116 = air.channel.get async [%arg8]  @channel_12[%c0_22, %c0_22] (%results_26[%c0_22, %arg7] [%c32_18, %c256_24] [%c1024_23, %c1_19]) {id = 13 : i32} : (memref<32x1024xi32, 1>)
      %117 = air.wait_all async [%116] 
      scf.yield %117 : !air.async.token
    }
    %48 = air.wait_all async 
    %49 = air.wait_all async [%async_token_27] 
    %50 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %49) -> (!air.async.token) {
      %116 = air.channel.get async [%arg8]  @channel_12[%c1_19, %c0_22] (%results_28[%c0_22, %arg7] [%c32_18, %c256_24] [%c1024_23, %c1_19]) {id = 14 : i32} : (memref<32x1024xi32, 1>)
      %117 = air.wait_all async [%116] 
      scf.yield %117 : !air.async.token
    }
    %51 = air.wait_all async 
    %52 = air.wait_all async [%async_token_29] 
    %53 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %52) -> (!air.async.token) {
      %116 = air.channel.get async [%arg8]  @channel_12[%c2_17, %c0_22] (%results_30[%c0_22, %arg7] [%c32_18, %c256_24] [%c1024_23, %c1_19]) {id = 15 : i32} : (memref<32x1024xi32, 1>)
      %117 = air.wait_all async [%116] 
      scf.yield %117 : !air.async.token
    }
    %54 = air.wait_all async [%async_token_31] 
    %55 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %54) -> (!air.async.token) {
      %116 = air.channel.get async [%arg8]  @channel_12[%c3_16, %c0_22] (%results_32[%c0_22, %arg7] [%c32_18, %c256_24] [%c1024_23, %c1_19]) {id = 16 : i32} : (memref<32x1024xi32, 1>)
      %117 = air.wait_all async [%116] 
      scf.yield %117 : !air.async.token
    }
    ...
    air.segment_terminator
  }
  air.launch_terminator
}
```
The pass analyzes any imperfectly nested `scf.for` loop which contains `air.channel.put` and `air.channel.get` operations and detemine, based on loop-carried async dependency paths passing through the channel operations, whether the loop is splittable into new asynchronously parallel `scf.for` loop nests, which only contain a single channel operation in its innermost iteration space.

This pass exposes channel operation optimization opportunities, where any perfect for-loop nests, parent to a channel operation, can be folded as new wrap-and-stride dimensions in its n-dimensional data layout rearrangement pattern. The isolation of perfect for-loop nests, and their subsequent folding into channel operations, is especially useful for AIE architecture, which features memtiles possessing large L2 memory space and high DMA streaming bandwidth, but no core to inference any control flow around its L2 memories.

In addition, the pass also eliminates any perfectly nested for loops surrounding any `air.herd`. The rationale is that the `air.herd` body will be outlined into compute tasks executed on AIE cores as well as finite-state machines of AIE tile DMA Buffer Descriptors, both of which are executing within infinite loops.

### air-loop-fusion

Optimizes the data movement around L2 memories by rearranging and potentially fusing perfect `scf.for` loop nests of `air.channel.put` and `air.channel.get`, which access the same L2 memref, into `scf.for` loop nest patterns mappable to a complex finite-state machine consisting of a multiple of AIE DMA Block Descriptors.

*Input IR:*
```
%30 = air.segment @segment_0 async  attributes {id = 2 : i32} {
  ...
  %async_token_25, %results_26 = air.execute -> (memref<32x1024xi32, 1>) {
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    air.execute_terminator %alloc : memref<32x1024xi32, 1>
  }
  %async_token_27, %results_28 = air.execute -> (memref<32x1024xi32, 1>) {
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    air.execute_terminator %alloc : memref<32x1024xi32, 1>
  }
  ...
  %34 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %async_token_25) -> (!air.async.token) {
    %73 = air.channel.get async [%arg8]  @channel_12[%c0_22, %c0_22] (%results_26[%c0_22, %arg7] [%c32_18, %c256_24] [%c1024_23, %c1_19]) {id = 13 : i32} : (memref<32x1024xi32, 1>)
    scf.yield %73 : !air.async.token
  }
  %35 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %async_token_27) -> (!air.async.token) {
    %73 = air.channel.get async [%arg8]  @channel_12[%c1_19, %c0_22] (%results_28[%c0_22, %arg7] [%c32_18, %c256_24] [%c1024_23, %c1_19]) {id = 14 : i32} : (memref<32x1024xi32, 1>)
    scf.yield %73 : !air.async.token
  }
  ...
  %43 = scf.for %arg7 = %c0_22 to %c128_20 step %c4_21 iter_args(%arg8 = %async_token_25) -> (!air.async.token) {
    %async_token_61, %results_62 = air.execute [%arg8] -> (index) {
      %74 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
      air.execute_terminator %74 : index
    }
    %73 = air.channel.put async [%async_token_61]  @channel_0[] (%results_26[%c0_22, %c0_22, %results_62] [%c4_21, %c32_18, %c8] [%c8, %c1024_23, %c1_19]) {id = 21 : i32} : (memref<32x1024xi32, 1>)
    scf.yield %73 : !air.async.token
  }
  %44 = scf.for %arg7 = %c0_22 to %c128_20 step %c4_21 iter_args(%arg8 = %async_token_27) -> (!air.async.token) {
    %async_token_61, %results_62 = air.execute [%arg8] -> (index) {
      %74 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg7]
      air.execute_terminator %74 : index
    }
    %73 = air.channel.put async [%async_token_61]  @channel_1[] (%results_28[%c0_22, %c0_22, %results_62] [%c4_21, %c32_18, %c8] [%c8, %c1024_23, %c1_19]) {id = 22 : i32} : (memref<32x1024xi32, 1>)
    scf.yield %73 : !air.async.token
  }
  ...
  air.segment_terminator
}
```
The input IR consists of some L2 memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations, nested under some perfect `scf.for` loop nest. L2 memory operations have been encapsulated within `air.segment`.

*Output IR:*
```
%30 = air.segment @segment_0 async  attributes {id = 2 : i32} {
  ...
  %35 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %34) -> (!air.async.token) {
    ...
    %async_token_41, %results_42 = air.execute -> (memref<32x256xi32, 1>) {
      %alloc = memref.alloc() : memref<32x256xi32, 1>
      air.execute_terminator %alloc : memref<32x256xi32, 1>
    }
    %async_token_43, %results_44 = air.execute -> (memref<32x256xi32, 1>) {
      %alloc = memref.alloc() : memref<32x256xi32, 1>
      air.execute_terminator %alloc : memref<32x256xi32, 1>
    }
    ...
    %63 = air.channel.get async [%arg8]  @channel_12[%c3_16, %c0_22] (%results_42[%c0_22, %arg7] [%c32_18, %c256_24] [%c256_24, %c1_19]) {id = 16 : i32} : (memref<32x256xi32, 1>)
    %64 = scf.for %arg9 = %c0_22 to %c256_24 step %c32_18 iter_args(%arg10 = %63) -> (!air.async.token) {
      %76 = air.channel.put async [%arg10]  @channel_3[] (%results_42[%c0_22, %c0_22, %arg9] [%c4_21, %c32_18, %c8] [%c8, %c256_24, %c1_19]) {id = 24 : i32} : (memref<32x256xi32, 1>)
      scf.yield %76 : !air.async.token
    }
    %65 = air.channel.get async [%arg8]  @channel_12[%c0_22, %c0_22] (%results_44[%c0_22, %arg7] [%c32_18, %c256_24] [%c256_24, %c1_19]) {id = 13 : i32} : (memref<32x256xi32, 1>)
    %66 = scf.for %arg9 = %c0_22 to %c256_24 step %c32_18 iter_args(%arg10 = %65) -> (!air.async.token) {
      %76 = air.channel.put async [%arg10]  @channel_0[] (%results_44[%c0_22, %c0_22, %arg9] [%c4_21, %c32_18, %c8] [%c8, %c256_24, %c1_19]) {id = 21 : i32} : (memref<32x256xi32, 1>)
      scf.yield %76 : !air.async.token
    }
    ...
    %async_token_55 = air.execute {
      memref.dealloc %results_42 : memref<32x256xi32, 1>
    }
    %async_token_56 = air.execute {
      memref.dealloc %results_44 : memref<32x256xi32, 1>
    }
    ...
  }
  ...
  air.segment_terminator
}
```
The pass attempts to merge perfectly nested for loops, which iterate over `air.channel.put` or `air.channel.get` accessing the same L2 `memref`, into a single loop. This can greatly facilitate any downstream conversion to physical representations of memory operations around the `memref`, using only a sequence of low-level constructs such as Block Descriptors.

The pass also attempts to confine the lifetime of the target L2 `memref` by moving its allocation `memref.alloc()` and deallocation `memref.dealloc()` into the begin and end of the fused for loop body, in order to facilitate the downstream compilation.

Within the confined lifetime of the L2 `memref`, the pass also attempts to analyze its effective data access pattern at each time phase in its lifetime, to determine whether its size can be shrunk. If true, then the memref's allocation and deallocation is updated to a reduced size, to optimize for memory utilization.

After loop fusion, the compilation pass analyzes the data production and consuption pattern around the L2 memref, and reconstructs the loop-carried asynch dependency within the fused loop body. The newly constructed dependency edges represent the transition edges connecting the states (i.e. `air.channel.put` and `air.channel.get` operations) of the finite-state machine.

### air-label-scf-for-to-ping-pong

The analysis pass detects ping-pong buffering opportunities implied in a `scf.for` loop, based on inspection of the data production and consumption around a `memref` with a lifetime within its body. Upon successful detection of ping-pong buffering opportunities, the pass assigns an `unroll=2` attribute, as compiler flag for a downstream pass to transform the loop into explicitly representing the ping-pong schedule.

*Input IR:*
```
%30 = air.segment @segment_0 async  attributes {id = 2 : i32} {
  ...
  %35 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %34) -> (!air.async.token) {
    %async_token_37, %results_38 = air.execute -> (memref<256x32xi32, 1>) {
      %alloc = memref.alloc() : memref<256x32xi32, 1>
      air.execute_terminator %alloc : memref<256x32xi32, 1>
    }
    ...
    %59 = air.channel.get async [%arg8]  @channel_13[%c1_19, %c0_22] (%results_38[%arg7, %c0_22] [%c256_24, %c32_18] [%c32_18, %c1_19]) {id = 18 : i32} : (memref<256x32xi32, 1>)
    %60 = scf.for %arg9 = %c0_22 to %c256_24 step %c32_18 iter_args(%arg10 = %59) -> (!air.async.token) {
      %76 = air.channel.put async [%arg10]  @channel_5[] (%results_38[%c0_22, %arg9, %c0_22] [%c8, %c32_18, %c4_21] [%c4_21, %c32_18, %c1_19]) {id = 26 : i32} : (memref<256x32xi32, 1>)
      scf.yield %76 : !air.async.token
    }
    ...
    %async_token_53 = air.execute {
      memref.dealloc %results_38 : memref<256x32xi32, 1>
    }
    ...
    scf.yield %75 : !air.async.token
  }
  ...
  air.segment_terminator
}
```
The input IR consists of some memory references (`memrefs`) accessed by `air.channel.put/get` data movement operations and `linalg.generic` compute operations, nested under some `scf.for` loop. The `memref` allocation and deallocation are located within the for loop iteration space, if its lifetime is strictly scoped within each iteration of the for loop. The `scf.for` loop needs to have MLIR-AIR async interface, in order for any subsequent ping-pong transformation (`air-ping-pong-transform`) pass to successfully represent the ping-pong schedule via async tokens.

*Output IR:*
```
%30 = air.segment @segment_0 async  attributes {id = 2 : i32} {
  ...
  %35 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %34) -> (!air.async.token) {
    %async_token_37, %results_38 = air.execute -> (memref<256x32xi32, 1>) {
      %alloc = memref.alloc() {hoist_alloc = true} : memref<256x32xi32, 1>
      air.execute_terminator %alloc : memref<256x32xi32, 1>
    }
    ...
    %59 = air.channel.get async [%arg8]  @channel_13[%c1_19, %c0_22] (%results_38[%arg7, %c0_22] [%c256_24, %c32_18] [%c32_18, %c1_19]) {id = 18 : i32} : (memref<256x32xi32, 1>)
    %60 = scf.for %arg9 = %c0_22 to %c256_24 step %c32_18 iter_args(%arg10 = %59) -> (!air.async.token) {
      %76 = air.channel.put async [%arg10]  @channel_5[] (%results_38[%c0_22, %arg9, %c0_22] [%c8, %c32_18, %c4_21] [%c4_21, %c32_18, %c1_19]) {id = 26 : i32} : (memref<256x32xi32, 1>)
      scf.yield %76 : !air.async.token
    }
    ...
    %async_token_53 = air.execute {
      memref.dealloc %results_38 : memref<256x32xi32, 1>
    }
    ...
    scf.yield %75 : !air.async.token
  } {unroll = 2 : i32}
  ...
  air.segment_terminator
}
```
The pass analyzes the computational graph of an `scf.for` loop body, and detect whether its hardware schedule is ping-pong transformable. Currently, any `scf.for` loop with any `memref` whose lifetime is scoped within its iteration, is considered as ping-pong transformable.

### air-ping-pong-transform

Transforms the IR into explicitly representing a ping-pong buffered hardware schedule, via unrolling the `scf.for` loop by two, and explicitly representing a multiple of async threads passing through the `scf.for` loop body in parallel.

*Input IR:*
```
%30 = air.segment @segment_0 async  attributes {id = 2 : i32} {
  ...
  %35 = scf.for %arg7 = %c0_22 to %c1024_23 step %c256_24 iter_args(%arg8 = %34) -> (!air.async.token) {
    %async_token_37, %results_38 = air.execute -> (memref<256x32xi32, 1>) {
      %alloc = memref.alloc() {hoist_alloc = true} : memref<256x32xi32, 1>
      air.execute_terminator %alloc : memref<256x32xi32, 1>
    }
    ...
    %59 = air.channel.get async [%arg8]  @channel_13[%c1_19, %c0_22] (%results_38[%arg7, %c0_22] [%c256_24, %c32_18] [%c32_18, %c1_19]) {id = 18 : i32} : (memref<256x32xi32, 1>)
    %60 = scf.for %arg9 = %c0_22 to %c256_24 step %c32_18 iter_args(%arg10 = %59) -> (!air.async.token) {
      %76 = air.channel.put async [%arg10]  @channel_5[] (%results_38[%c0_22, %arg9, %c0_22] [%c8, %c32_18, %c4_21] [%c4_21, %c32_18, %c1_19]) {id = 26 : i32} : (memref<256x32xi32, 1>)
      scf.yield %76 : !air.async.token
    }
    ...
    %async_token_53 = air.execute {
      memref.dealloc %results_38 : memref<256x32xi32, 1>
    }
    ...
    scf.yield %75 : !air.async.token
  } {unroll = 2 : i32}
  ...
  air.segment_terminator
}
```
The input IR consists of some memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations, nested under some `scf.for` loop, possibly labelled with an attribute `unroll=2` which is generated by a prior pass (`air-label-scf-for-to-ping-pong`).

*Output IR:*
```
%30 = air.segment @segment_0 async  attributes {id = 2 : i32} {
  ...
  %async_token_40, %results_41 = air.execute [%async_token_38] -> (memref<256x32xi32, 1>) {
    %alloc = memref.alloc() : memref<256x32xi32, 1>
    air.execute_terminator %alloc : memref<256x32xi32, 1>
  }
  ...
  %async_token_54, %results_55 = air.execute [%async_token_52] -> (memref<256x32xi32, 1>) {
    %alloc = memref.alloc() : memref<256x32xi32, 1>
    air.execute_terminator %alloc : memref<256x32xi32, 1>
  }
  ...
  %35:4 = scf.for %arg7 = %c0_23 to %c1024_24 step %c512_16 iter_args(%arg8 = %async_token_54, %arg9 = %async_token_56, %arg10 = %async_token_56, %arg11 = %async_token_56) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
    %59 = air.channel.get async [%arg11, %arg8]  @channel_13[%c1_20, %c0_23] (%results_55[%arg7, %c0_23] [%c256_25, %c32_19] [%c32_19, %c1_20]) {id = 18 : i32} : (memref<256x32xi32, 1>)
    %60 = air.wait_all async [%arg10, %59] 
    %61 = scf.for %arg12 = %c0_23 to %c256_25 step %c32_19 iter_args(%arg13 = %60) -> (!air.async.token) {
      %111 = air.channel.put async [%arg13]  @channel_5[] (%results_55[%c0_23, %arg12, %c0_23] [%c8, %c32_19, %c4_22] [%c4_22, %c32_19, %c1_20]) {id = 26 : i32} : (memref<256x32xi32, 1>)
      scf.yield %111 : !air.async.token
    }
    ...
    %async_token_70 = air.execute [%arg11, %arg8] {
      memref.dealloc %results_55 : memref<256x32xi32, 1>
    }
    ...
    %85 = air.channel.get async [%80, %77, %74, %71, %68, %65, %62, %59, %arg9]  @channel_13[%c1_20, %c0_23] (%results_41[%84, %c0_23] [%c256_25, %c32_19] [%c32_19, %c1_20]) {id = 18 : i32} : (memref<256x32xi32, 1>)
    %86 = air.wait_all async [%83, %85] 
    %87 = scf.for %arg12 = %c0_23 to %c256_25 step %c32_19 iter_args(%arg13 = %86) -> (!air.async.token) {
      %111 = air.channel.put async [%arg13]  @channel_5[] (%results_41[%c0_23, %arg12, %c0_23] [%c8, %c32_19, %c4_22] [%c4_22, %c32_19, %c1_20]) {id = 26 : i32} : (memref<256x32xi32, 1>)
      scf.yield %111 : !air.async.token
    }
    ...
    %async_token_78 = air.execute [%80, %77, %74, %71, %68, %65, %62, %59, %arg9] {
      memref.dealloc %results_41 : memref<256x32xi32, 1>
    }
    ...
    %109 = air.wait_all async [%85, %87, %88, %90, %91, %93, %94, %96, %97, %99, %100, %102, %103, %105, %106, %108, %async_token_78, %async_token_79, %async_token_80, %async_token_81, %async_token_82, %async_token_83, %async_token_84, %async_token_85] 
    %110 = air.wait_all async [%85, %88, %91, %94, %97, %100, %103, %106] 
    scf.yield %83, %109, %109, %110 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
  }
  ...
  air.segment_terminator
}
```
The pass transforms the computational graph of an `scf.for` loop body---previously detected by `air-label-scf-for-to-ping-pong` as being ping-pong transformable---to use "ping-pong" buffers and channels for asynchronous execution and data transfer, aiming to reduce memory access latency and increase throughput by overlapping computation with data transfer and utilizing local memory more effectively.

The pass is first unrolled by a factor of two to generate explicit "ping" and "pong" `memrefs` and async events, as handles to explicitly represent the parallelism between "ping" data producers and "pong" data consumers, and vice versa.

The `scf.for` body is transformed into having a multiple of async tokens (`%async_token`) passed into its body via `iter_args` operands/arguments, and yielded and returned via `scf.yield`. Each of those tokens represents an async thread flowing through a path made of async dependency edges across the for-loop iterations.

### air-specialize-channel-wrap-and-stride

Canonicalizes `air.channel.put` and `air.channel.get` operations' `offsets`, `sizes` and `strides` list by fold any perfectly nested parent `scf.for` loop into the list, and removing any redundant wrap-and-stride dimensions.

*Input IR*
```
scf.for %arg2 = %c0 to %c128 step %c32 {
  air.channel.put  @channel_1[%c0, %c0] (%arg0[%arg2] [%c32] [%c1]) : (memref<128xf32>)
}
...
scf.for %arg2 = %c0 to %c128 step %c32 {
  scf.for %arg3 = %c0 to %c128 step %c32 {
    air.channel.get  @channel_2[%c0, %c0] (%arg1[%arg2, %arg3] [%c32, %c32] [%c128, %c1]) : (memref<128x128xf32>)
  }
}
```
The input IR contains data movement The input IR consists of some memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations, nested under some `scf.for` loop.

*Output IR*
```
air.channel.put  @channel_1[%c0, %c0] (%arg0[] [] []) : (memref<128xf32>)
...
air.channel.get  @channel_2[%c0, %c0] (%arg1[%c0, %c0, %c0, %c0] [%c4, %c4, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (memref<128x128xf32>)
```
The pass adjusts the wraps (`sizes`) and strides of the data movement operations into eliminating any perfectly nested parent `scf.for` loop nests, and transforming them as new highest dimensions of offsets, wraps and strides lists, where the lower bounds become additional offsets, trip counts become additional wraps, and step sizes are used to infer additional strides.

The pass also identifies and eliminates any redundant entries in the `offsets`, `sizes` and `strides` lists of the data movement operations, facilitating downstream passes which map those lists to hardware-constrained n-dimensional DMA Block Descriptors.

### air-collapse-herd

Transforms the shape of `air.herd` by attempting to collapse to occupy complete columns on AIE device.

*Input IR:*
```
air.herd tile (%x, %y) in (%sx=%c2, %sy=%c2) {
  %c0 = arith.constant 0 : index
  ...
}
```
The input IR has the L1 memory management and computation encapsulated within `air.herd`.

*Output IR:*
```
air.herd  tile (%arg0, %arg1) in (%arg2=%c1, %arg3=%c4) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %0 = arith.remsi %arg1, %c2 : index
  %1 = arith.divsi %arg1, %c2 : index
  ...
  air.herd_terminator
}
```
The pass attempts to collapse the `air.herd` to the left, attempting to occupy complete columns of AIE tiles. The attempt will stop if the number of tiles in `air.herd` exceeds the user provided `max-col-size` option.

### air-place-herds

Places `air.herd` within its parent `air.segment` using Greedy method; infers `air.segment` shape based on the shape and size of all `air.herd` operations within its body.

*Input IR:*
```
%31 = air.segment @segment_0 async  attributes {id = 2 : i32} {
  ...
  %52 = air.herd @herd_0 async  tile (%arg7, %arg8) in (%arg9=%c4_17, %arg10=%c4_17) attributes {id = 3 : i32}
```
The input IR has the L1 memory management and computation encapsulated within `air.herd`, and L2 memory management and `air.herd` encapsulated within `air.segment`.

*Output IR:*
```
%31 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
  ...
  %52 = air.herd @herd_0 async  tile (%arg7, %arg8) in (%arg9=%c4_17, %arg10=%c4_17) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
```
After the pass has been applied, both the `air.segment` and `air.herd` operations have additional attributes (`x_loc, x_size, y_loc, y_size` for the segment, and `x_loc, y_loc` for the herd). These new attributes specify explicit spatial placement and size information:

`x_loc` and `y_loc` denote the starting location of the segment or herd in a two-dimensional space (a grid of AIE tiles).
`x_size` and `y_size` specify the dimensions of the segment, indicating how much space the segment occupies in each dimension.
The pass is responsible for spatially placing `air.herd` opertaions onto a grid of AIE tiles using Greedy method.

### air-to-aie

Converts an input IR from MLIR-AIR dialect into MLIR-AIE dialect, for efficient mapping onto AIEs. Outlines (1) `air.herd` to `aie.cores` and `aie.mems`, (2) `air.segment` to `aie.memtiles`, (3) `air.channel.put/get` to `aie.dma_start`, `aie.dma_bd` and `aie.use_locks`, and (4) `memref.alloc` to `aie.buffer`.

#### Outline air.herd to aie.cores and aie.mems

*Input IR:*
```
air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<1024xi32>, memref<1024xi32> attributes { sym_name="herd1"} {
  %buf0 = memref.alloc() : memref<1024xi32, 2>
  %buf1 = memref.alloc() : memref<512xi32, 2>
  air.channel.get @channel_0[%tx, %ty] (%buf0[] [] []) {id = 2 : i32} : (memref<1024xi32, 2>)
  air.channel.put @channel_1[%tx, %ty] (%buf1[] [] []) {id = 3 : i32} : (memref<512xi32, 2>)
  memref.dealloc %buf0 : memref<1024xi32, 2>
  memref.dealloc %buf1 : memref<512xi32, 2>
  air.herd_terminator
}
```
The input MLIR-AIR dialect code specifies the spatial compute across a rectangular plot of AIE cores using `air.herd`.

*Output IR:*
```
%buf1 = aie.buffer(%tile_2_3) {sym_name = "buf1"} : memref<1024xi32, 2> 
%buf0 = aie.buffer(%tile_2_3) {sym_name = "buf0"} : memref<512xi32, 2> 
...
%mem_2_3 = aie.mem(%tile_2_3) {
  %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
^bb1:  // 2 preds: ^bb0, ^bb1
  aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
  aie.dma_bd(%buf1 : memref<1024xi32, 2>, 0, 1024)
  aie.use_lock(%lock_2_3_0, Release, 1)
  aie.next_bd ^bb1
^bb2:  // pred: ^bb3
  aie.end
^bb3:  // pred: ^bb0
  %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
^bb4:  // 2 preds: ^bb3, ^bb4
  aie.use_lock(%lock_2_3_2, AcquireGreaterEqual, 1)
  aie.dma_bd(%buf0 : memref<512xi32, 2>, 0, 512)
  aie.use_lock(%lock_2_3_1, Release, 1)
  aie.next_bd ^bb4
}
%core_2_3 = aie.core(%tile_2_3) {
  cf.br ^bb1
^bb1:  // pred: ^bb0
  cf.br ^bb2
^bb2:  // pred: ^bb1
  aie.use_lock(%lock_2_3_1, AcquireGreaterEqual, 1)
  aie.use_lock(%lock_2_3_0, AcquireGreaterEqual, 1)
  aie.use_lock(%lock_2_3, Release, 1)
  aie.use_lock(%lock_2_3_2, Release, 1)
  aie.end
} {elf_file = "herd1_core_2_3.elf"}
```
The output IR outlines the computation in each AIE core using `aie.core`, and data movement to/from the core using `aie.dma_bd` within `aie.mem`. Memref is allocated to `aie.buffer`. Each AIE tile's `aie.core` and `aie.mem` synchronize using `aie.use_lock`. Each core links to a unique `elf` file. The AIE core code includes explicit control flow for data movement and computation (`cf.br`, `scf.for` loops) iterating over data chunks.

#### Outline air.segment to aie.memtile

*Input IR:*
```
air.segment @segment0 {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  %memtile0 = memref.alloc() : memref<1024xi32, 1>
  air.channel.get @channel_2[] (%memtile0[] [] []) {id = 2 : i32} : (memref<1024xi32, 1>)
  air.channel.put @channel_3[] (%memtile0[] [] []) {id = 3 : i32} : (memref<1024xi32, 1>)
  memref.dealloc %memtile0 : memref<1024xi32, 1>
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) attributes { sym_name="herd4"} {
    ...
    air.herd_terminator
  }
  %memtile1 = memref.alloc() : memref<1024xi32, 1>
  air.channel.get @channel_4[] (%memtile1[] [] []) {id = 6 : i32} : (memref<1024xi32, 1>)
  air.channel.put @channel_5[] (%memtile1[] [] []) {id = 7 : i32} : (memref<1024xi32, 1>)
  memref.dealloc %memtile1 : memref<1024xi32, 1>
  air.segment_terminator
}
```
The input MLIR-AIR dialect code encapsulates any L2 memory allocation, deallocation, and data movement within `air.segment`.

*Output IR:*
```
%buf2 = aie.buffer(%tile_2_1) {sym_name = "buf2"} : memref<1024xi32, 1> 
%buf1 = aie.buffer(%tile_2_1) {sym_name = "buf1"} : memref<1024xi32, 1> 
...
%memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
  %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
^bb1:  // 2 preds: ^bb0, ^bb1
  aie.use_lock(%lock_2_1_1, AcquireGreaterEqual, 1)
  aie.dma_bd(%buf2 : memref<1024xi32, 1>, 0, 1024)
  aie.use_lock(%lock_2_1_2, Release, 1)
  aie.next_bd ^bb1
^bb2:  // pred: ^bb3
  aie.end
^bb3:  // pred: ^bb5
  %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
^bb4:  // 2 preds: ^bb3, ^bb4
  aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
  aie.dma_bd(%buf1 : memref<1024xi32, 1>, 0, 1024)
  aie.use_lock(%lock_2_1_0, Release, 1)
  aie.next_bd ^bb4
^bb5:  // pred: ^bb7
  %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
^bb6:  // 2 preds: ^bb5, ^bb6
  aie.use_lock(%lock_2_1_2, AcquireGreaterEqual, 1)
  aie.dma_bd(%buf2 : memref<1024xi32, 1>, 0, 1024)
  aie.use_lock(%lock_2_1_1, Release, 1)
  aie.next_bd ^bb6
^bb7:  // pred: ^bb0
  %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb5, repeat_count = 1)
^bb8:  // 2 preds: ^bb7, ^bb8
  aie.use_lock(%lock_2_1_0, AcquireGreaterEqual, 1)
  aie.dma_bd(%buf1 : memref<1024xi32, 1>, 0, 1024)
  aie.use_lock(%lock_2_1, Release, 1)
  aie.next_bd ^bb8
}
```
The output IR outlines the data movement to/from each memtile using `aie.memtile_dma`. Multiple `aie.dma_bd` within the same AIE tile synchronize using `aie.dma_bd`.

#### Outline AIE circuit-switched streaming interconnects using aie.flow.

*Input IR:*
```
func.func @func4(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
  ...
  air.segment @segment0  {
    ...
    air.channel.put  @channel_3[] (%alloc[] [] []) {id = 3 : i32} : (memref<1024xi32, 1>)
    ...
    air.herd @herd4  tile (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_1) attributes {x_loc = 2 : i32, y_loc = 3 : i32} {
      ...
      air.channel.get  @channel_3[%arg2, %arg3] (%alloc_3[] [] []) {id = 4 : i32} : (memref<1024xi32, 2>)
      ...
      air.herd_terminator
    }
    ...
    air.segment_terminator
  }
  ...
  return
}
```
The input IR represents data movement across memory spaces using `air.channel.put/get`, paired using channel symbols.

*Output IR:*
```
aie.flow(%tile_2_1, DMA : 0, %tile_2_3, DMA : 0)
```
The output IR outlines circuit-switched streaming dataflow using `aie.flow`, connecting the source and destination DMA channels.

#### Generate aie.shim_dma_allocation as handle to runtime operations.

*Input IR:*
```
func.func @func4(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) -> () {
  ...
  air.channel.put @channel_2[] (%arg0[] [] []) {id = 1 : i32} : (memref<1024xi32>)
  air.segment @segment0 {
    ...
    air.channel.get @channel_2[] (%memtile0[] [] []) {id = 2 : i32} : (memref<1024xi32, 1>)
    ...
    air.channel.put @channel_5[] (%memtile1[] [] []) {id = 7 : i32} : (memref<1024xi32, 1>)
    ...
    air.segment_terminator
  }
  ...
  air.channel.get @channel_5[] (%arg1[] [] []) {id = 8 : i32} : (memref<1024xi32>)
  return
}
```
The input IR represent data movements to/from external memory using `air.channel.put/get` on L3 memref.

*Output IR:*
```
aie.device(xcve2802) {
  ...
  aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 2)
  memref.global "public" @airMemcpyId7 : memref<1024xi32, 1>
  aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 2)
  memref.global "public" @airMemcpyId2 : memref<1024xi32, 1>
  ...
} {sym_name = "segment0"}
...
func.func @func4(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
  ...
  air.channel.put  @channel_2[] (%arg0[] [] []) {id = 1 : i32, metadata = @airMemcpyId2} : (memref<1024xi32>)
  ...
  air.channel.get  @channel_5[] (%arg1[] [] []) {id = 8 : i32, metadata = @airMemcpyId7} : (memref<1024xi32>)
  ...
  return
}
```
The output IR generates `aie.shim_dma_allocation` as handle to link with the runtime code using `metadata` attribute.

### air-to-std

Converts the MLIR-AIR dialect code into AIRRt dialect which represents the runtime code dispatching the program by pushing and pulling data from the AIE SHIM tile DMAs.

*Input IR:*
```
module {
  aie.device(npu) {
    ...
    aie.shim_dma_allocation @airMemcpyId78(S2MM, 0, 0)
    memref.global "public" @airMemcpyId78 : memref<32x128xi32, 1>
    ...
    aie.shim_dma_allocation @airMemcpyId19(MM2S, 0, 0)
    memref.global "public" @airMemcpyId19 : memref<32x256xi32, 1>
    ...
    aie.shim_dma_allocation @airMemcpyId15(MM2S, 0, 2)
    memref.global "public" @airMemcpyId15 : memref<256x32xi32, 1>
    ...
  } {sym_name = "segment_0"}
  ...
  func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
    ...
    %0 = air.launch async [%async_token_0, %async_token_3, %async_token_6] (%arg0, %arg1) in (%arg2=%c4, %arg3=%c4) args(%arg4=%results_5, %arg5=%results, %arg6=%results_2) : memref<512x512xi32>, memref<512x1024xi32>, memref<1024x512xi32> attributes {id = 1 : i32} {
      ...
      %6 = air.channel.put async [%5]  @channel_12[%c0_8, %c0_8] (%arg5[%c0_8, %1, %c0_8] [%c4_7, %c32, %c256] [%c256, %c1024, %c1]) {id = 1 : i32, metadata = @airMemcpyId19} : (memref<512x1024xi32>)
      ...
      %18 = air.channel.put async [%17]  @channel_13[%c0_8, %c0_8] (%arg6[%c0_8, %13] [%c1024, %c32] [%c512, %c1]) {id = 5 : i32, metadata = @airMemcpyId15} : (memref<1024x512xi32>)
      ...
      %26 = air.channel.get async [%25, %async_token_9]  @channel_14[%c0_8, %c0_8] (%arg4[%1, %results_10] [%c32, %c128] [%c512, %c1]) {id = 9 : i32, metadata = @airMemcpyId78} : (memref<512x512xi32>)
      ...
      air.launch_terminator
    }
    return
  }
}
```
The input IR contains some `air.channel.put` and `air.channel.get` memory operations on external (L3) memory references (`memrefs`), optionally encapsulated within `air.launch`. Those memory operations are annotated by `metadata` attribute which maintains a symbolic link to an AIE SHIM DMA channel.

*Output IR:*
```
module {
  aie.device(npu) {
    ...
    aie.shim_dma_allocation @airMemcpyId78(S2MM, 0, 0)
    memref.global "public" @airMemcpyId78 : memref<32x128xi32, 1>
    ...
    aie.shim_dma_allocation @airMemcpyId19(MM2S, 0, 0)
    memref.global "public" @airMemcpyId19 : memref<32x256xi32, 1>
    ...
    aie.shim_dma_allocation @airMemcpyId15(MM2S, 0, 2)
    memref.global "public" @airMemcpyId15 : memref<256x32xi32, 1>
    ...
  } {sym_name = "segment_0"}
  ...
  func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
    ...
    %9 = airrt.wait_all %8, %5, %2 : !airrt.event
    affine.for %arg0 = 0 to 4 {
      affine.for %arg1 = 0 to 4 {
        ...
        %25 = airrt.dma_memcpy_nd(%c17_i32, %15, %16, %0[%c0_i64, %17, %18, %19], [%c1_i64, %22, %23, %24], [%c0_i64, %20, %21]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x1024xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %26 = airrt.wait_all : !airrt.event
        ...
        %74 = airrt.dma_memcpy_nd(%c13_i32, %67, %68, %3[%c0_i64_13, %c0_i64_13, %69, %70], [%c1_i64_14, %c1_i64_14, %72, %73], [%c0_i64_13, %c0_i64_13, %71]) {metadata = @airMemcpyId15} : (i32, i64, i64, memref<1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %75 = airrt.wait_all : !airrt.event
        ...
        %111 = airrt.dma_memcpy_nd(%c78_i32, %104, %105, %6[%c0_i64_24, %c0_i64_24, %106, %107], [%c1_i64_25, %c1_i64_25, %109, %110], [%c0_i64_24, %c0_i64_24, %108]) {metadata = @airMemcpyId78} : (i32, i64, i64, memref<512x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        ...
      }
    } {affine_opt_label = "tiling"}
    return
  }
}
```
After the conversion, the control flow, synchronization and DMA operations responsible for external memory (L3) copies via AIE SHIM DMAs are retained, while other operations are discarded. They are replaced with the AIRRt dialect operations which are dedicated to the representation of runtime operations, including explicit memory allocation (`airrt.alloc`), SHIM DMA data movement (`airrt.dma_memcpy_nd`), and synchronization (`airrt.wait_all`).

The use of `airrt.dma_memcpy_nd` after transformation shows detailed control over data movement, specifying the n-dimensional offsets, wraps and strides of data layout to be moved through SHIM DMA. This level of control is essential for optimizing data transfers between the AIE accelerator and external memory, where bandwidth is a limiting factor.

The control flow is represented with `affine` and `scf` loops, either retained from the input IR, or lowered from the `air.launch` iteration space.

The transformation includes explicit synchronization points (`airrt.wait_all`), which are crucial for managing dependencies between parallel operations, ensuring correct execution order without unnecessary stalls.

### affine-loop-opt

Transforms the IR by optimizing `affine` loop nests. `affine` loop nests are loops that iterate over multi-dimensional arrays in a manner that can be described using `affine` transformations.

*Input IR:*
```
func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
  ...
  affine.for %arg0 = 0 to 4 {
    affine.for %arg1 = 0 to 4 {
      ...
      %25 = airrt.dma_memcpy_nd(%c17_i32, %15, %16, %0[%c0_i64, %17, %18, %19], [%c1_i64, %22, %23, %24], [%c0_i64, %20, %21]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x1024xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      ...
      %74 = airrt.dma_memcpy_nd(%c13_i32, %67, %68, %3[%c0_i64_13, %c0_i64_13, %69, %70], [%c1_i64_14, %c1_i64_14, %72, %73], [%c0_i64_13, %c0_i64_13, %71]) {metadata = @airMemcpyId15} : (i32, i64, i64, memref<1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      ...
      %111 = airrt.dma_memcpy_nd(%c78_i32, %104, %105, %6[%c0_i64_24, %c0_i64_24, %106, %107], [%c1_i64_25, %c1_i64_25, %109, %110], [%c0_i64_24, %c0_i64_24, %108]) {metadata = @airMemcpyId78} : (i32, i64, i64, memref<512x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      ...
    }
  } {affine_opt_label = "tiling"}
  return
}
```
The input AIRRt dialect code represents the runtime program for the AIE accelerator, where the control flow is represented as `affine` and `scf` loops.

*Output IR:*
```
func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
  ...
  affine.for %arg0 = 0 to 4 step 4 {
    affine.for %arg1 = 0 to 4 step 4 {
      affine.for %arg2 = affine_map<(d0) -> (d0)>(%arg0) to affine_map<(d0) -> (d0 + 4)>(%arg0) {
        affine.for %arg3 = affine_map<(d0) -> (d0)>(%arg1) to affine_map<(d0) -> (d0 + 4)>(%arg1) {
          ...
          %25 = airrt.dma_memcpy_nd(%c17_i32, %15, %16, %0[%c0_i64, %17, %18, %19], [%c1_i64, %22, %23, %24], [%c0_i64, %20, %21]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x1024xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
          ...
          %74 = airrt.dma_memcpy_nd(%c13_i32, %67, %68, %3[%c0_i64_13, %c0_i64_13, %69, %70], [%c1_i64_14, %c1_i64_14, %72, %73], [%c0_i64_13, %c0_i64_13, %71]) {metadata = @airMemcpyId15} : (i32, i64, i64, memref<1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
          ...
          %111 = airrt.dma_memcpy_nd(%c78_i32, %104, %105, %6[%c0_i64_24, %c0_i64_24, %106, %107], [%c1_i64_25, %c1_i64_25, %109, %110], [%c0_i64_24, %c0_i64_24, %108]) {metadata = @airMemcpyId78} : (i32, i64, i64, memref<512x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
          ...
        }
      }
    }
  } {affine_opt_label = ""}
  return
}
```
The most notable transformation applied by the pass is loop tiling on `affine` loop nests containing `airrt.dma_memcpy_nd` operations in innermost loop. The pass attempts to tile the loop nest by user-provided factors (option `affine-opt-tile-sizes`). The tiled loops are then unrolled by a downstream pass (`air-unroll-outer-affine-loops`) to give an unrolled sequence of SHIM DMA Block Descriptors.

### air-unroll-outer-affine-loops
Unrolls the outermost dimensions in `affine` loop nests of the AIRRt runtime code.

*Input IR:*
```
func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
  ...
  affine.for %arg0 = 0 to 4 step 4 {
    affine.for %arg1 = 0 to 4 step 4 {
      affine.for %arg2 = affine_map<(d0) -> (d0)>(%arg0) to affine_map<(d0) -> (d0 + 4)>(%arg0) {
        affine.for %arg3 = affine_map<(d0) -> (d0)>(%arg1) to affine_map<(d0) -> (d0 + 4)>(%arg1) {
          ...
          %25 = airrt.dma_memcpy_nd(%c17_i32, %15, %16, %0[%c0_i64, %17, %18, %19], [%c1_i64, %22, %23, %24], [%c0_i64, %20, %21]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x1024xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
          ...
          %74 = airrt.dma_memcpy_nd(%c13_i32, %67, %68, %3[%c0_i64_13, %c0_i64_13, %69, %70], [%c1_i64_14, %c1_i64_14, %72, %73], [%c0_i64_13, %c0_i64_13, %71]) {metadata = @airMemcpyId15} : (i32, i64, i64, memref<1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
          ...
          %111 = airrt.dma_memcpy_nd(%c78_i32, %104, %105, %6[%c0_i64_24, %c0_i64_24, %106, %107], [%c1_i64_25, %c1_i64_25, %109, %110], [%c0_i64_24, %c0_i64_24, %108]) {metadata = @airMemcpyId78} : (i32, i64, i64, memref<512x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
          ...
        }
      }
    }
  } {affine_opt_label = ""}
  return
}
```
The input IR contains nested `affine.for` loops, previously tiled by a prior pass (`affine-loop-opt`) into having a desirable number of dimensions. Inside the loops, there are operations like `airrt.segment_load`, `arith.constant`, `arith.index_cast`, `airrt.dma_memcpy_nd`, and `airrt.wait_all`, representing program memory operations, arithmetic operations, and synchronization points. These operations are related to loading the binaries for AIE tiles, performing computation, and ensuring data is moved efficiently between the AIE device and external memory.

*Output IR:*
```
func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
  ...
  affine.for %arg0 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0 + 4)>(%c0) {
    affine.for %arg1 = affine_map<(d0) -> (d0)>(%c0_0) to affine_map<(d0) -> (d0 + 4)>(%c0_0) {
      ...
      %25 = airrt.dma_memcpy_nd(%c17_i32, %15, %16, %0[%c0_i64, %17, %18, %19], [%c1_i64, %22, %23, %24], [%c0_i64, %20, %21]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x1024xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      ...
      %74 = airrt.dma_memcpy_nd(%c13_i32, %67, %68, %3[%c0_i64_15, %c0_i64_15, %69, %70], [%c1_i64_16, %c1_i64_16, %72, %73], [%c0_i64_15, %c0_i64_15, %71]) {metadata = @airMemcpyId15} : (i32, i64, i64, memref<1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      ...
      %111 = airrt.dma_memcpy_nd(%c78_i32, %104, %105, %6[%c0_i64_26, %c0_i64_26, %106, %107], [%c1_i64_27, %c1_i64_27, %109, %110], [%c0_i64_26, %c0_i64_26, %108]) {metadata = @airMemcpyId78} : (i32, i64, i64, memref<512x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      ...
    }
  }
  return
}
```
This pass works together with a prior pass (`affine-loop-opt`) to generate a desirable number of unrolled instances of `airrt.dma_memcpy_nd` operations per AIE SHIM DMA channel, so that when those operations are lowered to AIE DMA BDs---plus the insertion of DMA BD reprograming synchronization points---by a downstream pass (`airrt-to-npu`), they do not violate the AIE DMA BD count limitations.

### airrt-to-npu
Converts the runtime program, described in AIRRt dialect, into instruction sequence specific to the SHIM DMA controllers on Ryzen AI platform.

*Input IR:*
```
module {
  aie.device(npu) {
    ...
    aie.shim_dma_allocation @airMemcpyId78(S2MM, 0, 0)
    memref.global "public" @airMemcpyId78 : memref<32x128xi32, 1>
    ...
    aie.shim_dma_allocation @airMemcpyId19(MM2S, 0, 0)
    memref.global "public" @airMemcpyId19 : memref<32x256xi32, 1>
    ...
    aie.shim_dma_allocation @airMemcpyId15(MM2S, 0, 2)
    memref.global "public" @airMemcpyId15 : memref<256x32xi32, 1>
    ...
  } {sym_name = "segment_0"}
  ...
  func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
    ...
    affine.for %arg0 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0 + 4)>(%c0) {
      affine.for %arg1 = affine_map<(d0) -> (d0)>(%c0_0) to affine_map<(d0) -> (d0 + 4)>(%c0_0) {
        ...
        %25 = airrt.dma_memcpy_nd(%c17_i32, %15, %16, %0[%c0_i64, %17, %18, %19], [%c1_i64, %22, %23, %24], [%c0_i64, %20, %21]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x1024xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        ...
        %74 = airrt.dma_memcpy_nd(%c13_i32, %67, %68, %3[%c0_i64_15, %c0_i64_15, %69, %70], [%c1_i64_16, %c1_i64_16, %72, %73], [%c0_i64_15, %c0_i64_15, %71]) {metadata = @airMemcpyId15} : (i32, i64, i64, memref<1024x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        ...
        %111 = airrt.dma_memcpy_nd(%c78_i32, %104, %105, %6[%c0_i64_26, %c0_i64_26, %106, %107], [%c1_i64_27, %c1_i64_27, %109, %110], [%c0_i64_26, %c0_i64_26, %108]) {metadata = @airMemcpyId78} : (i32, i64, i64, memref<512x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        ...
      }
    }
    return
  }
}
```
The input IR contains some L3 memory operations (`airrt.dma_memcpy_nd`) optionally nested within some nested `affine.for` loops.

*Output IR:*
```
module {
  aie.device(npu) {
    ...
    aie.shim_dma_allocation @airMemcpyId78(S2MM, 0, 0)
    memref.global "public" @airMemcpyId78 : memref<32x128xi32, 1>
    ...
    aie.shim_dma_allocation @airMemcpyId19(MM2S, 0, 0)
    memref.global "public" @airMemcpyId19 : memref<32x256xi32, 1>
    ...
    aie.shim_dma_allocation @airMemcpyId15(MM2S, 0, 2)
    memref.global "public" @airMemcpyId15 : memref<256x32xi32, 1>
    ...
    func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32() {
      ...
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][4, 4, 32, 256][0, 256, 1024]) {id = 0 : i64, metadata = @airMemcpyId19} : memref<512x1024xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 128, 0][4, 4, 32, 256][0, 256, 1024]) {id = 1 : i64, metadata = @airMemcpyId19} : memref<512x1024xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 256, 0][4, 4, 32, 256][0, 256, 1024]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x1024xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 384, 0][4, 4, 32, 256][0, 256, 1024]) {id = 3 : i64, metadata = @airMemcpyId19} : memref<512x1024xi32>
      ...
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 2, 512, 32][128, 262144, 512]) {id = 0 : i64, metadata = @airMemcpyId15} : memref<1024x512xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 2, 512, 32][128, 262144, 512]) {id = 1 : i64, metadata = @airMemcpyId15} : memref<1024x512xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 2, 512, 32][128, 262144, 512]) {id = 2 : i64, metadata = @airMemcpyId15} : memref<1024x512xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 2, 512, 32][128, 262144, 512]) {id = 3 : i64, metadata = @airMemcpyId15} : memref<1024x512xi32>
      ...
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][4, 4, 32, 128][65536, 128, 512]) {id = 8 : i64, metadata = @airMemcpyId78} : memref<512x512xi32>
      ...
      return
    }
  } {sym_name = "segment_0"}
}
```
The output is significantly simplified and optimized compared to the input. It focuses on the data movement instructions on the AIE SHIM DMA controller (`aiex.npu.dma_memcpy_nd`) driving the data movement between the AIE accelerator and external memory via the SHIM DMA.

The pass attempts to eliminate any `affine.for` loops by performing a sequence of loop transformations, aiming for folding those loop nests into additional wrap-and-stride data access patterns in the DMA BDs if possible. When the attempt fails, falls back to unrolling the loops into generating longer instruction sequence.

The transformed code introduces `aiex.npu.sync` operations used to reprogram all DMA Block Descriptors in a SHIM DMA.

The function signature in the output code (e.g. `func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32(%arg0: memref<512x1024xi32>, %arg1: memref<1024x512xi32>, %arg2: memref<512x512xi32>)`) takes the L3 `memrefs` to external memory as input.
