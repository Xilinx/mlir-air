# Bfloat16 GEMM on Ryzen AI: A Case Study for MLIR-AIR compilation pipeline.

## MLIR-AIR Compilation Recipe
    
    ################################################
    ## Binding scf.parallel to air hierarchies
    ################################################

        "buffer-results-to-out-params",
        "air-linalg-to-func{link-with=mm.o}",
        "air-par-to-herd{depth=1}",
        "air-par-to-launch{has-air-segment=true}",
        "air-copy-to-dma",
        "canonicalize", "cse",
    
    ###############################################
    # Extract event dependency and optimize schedule
    ###############################################

        "air-dependency",
        "air-dependency-schedule-opt",
        "air-specialize-dma-broadcast",
        "air-dma-to-channel",
        "canonicalize", "cse",
        "air-dependency-canonicalize",
        "canonicalize", "cse",
        'func.func(air-split-l2-memref)',
        "air-isolate-async-dma-loop-nests",
        "canonicalize", "cse",
        "func.func(air-loop-fusion)",
        "air-label-scf-for-to-ping-pong",
        "air-ping-pong-transform{keep-memref-dealloc=true}",
        "canonicalize", "cse",
        "air-specialize-channel-wrap-and-stride",
        "canonicalize", "cse",
    
    ################################################
    ## Place herd to segment
    ################################################

        "func.func(air-collapse-herd{max-col-size=4})",
        'canonicalize', 'cse',
        "air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0}",
        'canonicalize', 'cse',
        'func.func(air-renumber-dma)'
    
    ################################################
    ## MLIR-AIR to MLIR-AIE
    ################################################
    
        'canonicalize', 'cse',
        'air-to-aie{row-offset=2 col-offset=0 device=ipu emit-while-loop=true}',
        'canonicalize',
    
    ################################################
    ## MLIR-AIR runtime lowering
    ################################################

        'air-to-std',
        'canonicalize',
        'symbol-dce',
        'func.func(affine-loop-opt{affine-opt-tile-sizes=4,4})',
        'func.func(air-unroll-outer-affine-loops{depth=2})',
        'affine-expand-index-ops',
        'airrt-to-ipu',
        'canonicalize',

## Overview

|Compilation Stage |Passes |Description | Assumption on input IR |
|:--- |:--- |:--- |:--- |
|Binding MLIR operations to MLIR-AIR    |   <br> <ul><li>`air-linalg-to-func{link-with=mm.o}`</li><li>`air-par-to-herd{depth=1}`</li><li>`air-par-to-launch{has-air-segment=true}`</li><li>`air-copy-to-dma`</li></ul>    |   Binding parallelizable loops to `air` hierarchies; binding data movement operations to `air.dma_memcpy_nd` operations; binding linear algebra compute operations with link to AIE core kernel. | |
|Extract asynchronous event dependency and construct a task graph    |   <br> <ul><li>`air-dependency`</li><li>`air-dependency-canonicalize`</li></ul>    |   Construction of asynchronous task graph, as an explicit representation of the asynchronous concurrency in the hardware schedule. | |
|Broadcasting data movement    |   <br> <ul><li>`air-dependency-schedule-opt`</li><li>`air-specialize-dma-broadcast`</li></ul>    |   Detection and lowering of broadcasting data movement to map to circuit-routed streaming interconnects. |
|Generating half-dma operations mappable to AIE DMA Block Descriptors    |   <br> <ul><li>`air-dma-to-channel`</li></ul>    |   Lowering synchronous or asynchronous `air.dma_memcpy_nd` operations to `air.channel.put` or `air.channel.get` operations representing half-dma data sends and receives. | |
|Memtile (L2) buffer allocation optimization    |   <br> <ul><li>`func.func(air-split-l2-memref)`</li></ul>    |   Tiling L2 memrefs based on parallelizable data movements, explicitly represented via `scf.parallel` or `air.channel.put/get` operations, in order to maximize memtile bandwidth utilization. | |
|Memtile (L2) Block Descriptor lowering and optimization    |   <br> <ul><li>`air-isolate-async-dma-loop-nests`</li><li>`func.func(air-loop-fusion)`</li><li>`air-specialize-channel-wrap-and-stride`</li></ul>    |   Lowering L2 control flow program into finite-state machines made of Block Descriptors as states. | |
|Inferring double buffering patterns    |   <br> <ul><li>`air-label-scf-for-to-ping-pong`</li><li>`air-ping-pong-transform{keep-memref-dealloc=true}`</li></ul>    |   Detecting and lowering double buffering opportunities by analyzing data production and consumption patterns to a `memref` within an `scf.for` loop; explicitly represent the multiple asynchronous threads traversing through the loop. | |
|Place herd to segment    |   <br> <ul><li>`func.func(air-collapse-herd{max-col-size=4})`</li><li>`air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0}`</li><li>`func.func(air-renumber-dma)`</li></ul>    |   Reshaping and placing `air.herd` onto `air.segment`; inferring `air.segment` shape and size. | |
|MLIR-AIR to MLIR-AIE    |   <br> <ul><li>`func.func(air-renumber-dma)`</li><li>`air-to-aie{row-offset=2 col-offset=0 device=ipu emit-while-loop=true}`</li></ul>    |   Converting to MLIR-AIE dialect. Clone the `func.func` op, where one copy lowers to the circuit design to be mapped onto AIE tiles, and the other copy lowers to LX6 control program; outline `air.herd` body into `aie.core` kernel; materialize asynchronous `air.channel.put/get` into dma block descriptors and `aie.lock`. | |
|MLIR-AIR runtime lowering    |   <br> <ul><li>`air-to-std`</li><li>`func.func(affine-loop-opt{affine-opt-tile-sizes=4,4})`</li><li>`func.func(air-unroll-outer-affine-loops{depth=2})`</li><li>`airrt-to-ipu`</li></ul>    |   Converting the control code via AIRRt and AIEX.IPU dialect to LX6 instruction sequence. | |
||||||

## MLIR-AIR Passes

`air-par-to-herd`

Converts parallel computations, represented by `scf.parallel` or `scf.forall`, into a more optimized, hardware-specific form called `air.herd`. This transformation is targeted towards accelerating the matrix multiplication problem, by partitioning the proglem into strictly spatial concurrent threads represented by the iteration space of `air.herd`.

*Input IR:*
The input IR sets up a matrix multiplication operation, with input and output data references to memory via `memref` values. It includes MLIR operations which specify sub-views to memory and subdivide the work into smaller chunks that can be processed in parallel or in a loop, for performance optimization.

It features nested `scf.parallel` and `scf.for` loops to iterate over matrix elements, creating subviews and temporary buffers (`memref.alloc`) for storing intermediate results in each AIE tile's local (L1) memory. Inside these loops, `memref.copy` and `linalg.copy` operations are used to move data between different parts of memory, indicating a complex memory management pattern aimed at optimizing data locality and access patterns for the matrix multiplication.

*Output IR:*
The transformed code replaces certain `scf.parallel` loop structures in the loop nest---specified by `depth` option---with the air.herd operation, where the `air.herd` encapsulates the parallelism in memory management and computation (like loading, computing, and storing results) that is common in matrix multiplications.

By consolidating the `scf.parallel` body with the `isolateFromAbove` `air.herd` interface, the `air.herd` operation reduces the overhead associated with loop management and memory operations within the parent region.

The L1 memory allocation, deallocation, and copying operations that prepare data for the `air.herd` operation are retained within its body. This ensures that the memory management within each of its parallel thread is still handled explicitly, to maintain control over data layout and access patterns.

`air-par-to-launch`

Converts parallel computations, represented by `scf.parallel` or `scf.forall`, into `air.launch` construct. This transformation is targeted towards the dispatching of the already parallelized computations---represented by `air.herd`---in a more structured and *potentially* parallelized manner that is better suited for the dynamic launching of program iterations to reuse hardware configurations by the host.

*Input IR:*
The input IR uses nested loops (`scf.parallel` and `scf.for`) to iterate over the chunks of the matrices and perform computations in a tiled manner, with `scf.parallel` representing parallelizable chunks and `scf.for` representing strictly sqeuential chunks due to loop-carried dependency.

Inside the parallel and for-loop constructs, there are detailed memory management operations (`memref.alloc`, `memref.copy`) and computational kernels (encapsulated in the `air.herd` operation).

*Output IR:*
The transformed code introduces `air.launch` as the top-level construct, indicating the start of a parallelizable hardware-accelerated computation. Within `air.launch`, an `air.segment` construct is also optionally generated---controlled by option `has-air-segment`---representing a partition of the program, consisting of single or multiple `air.herd` and L2 memory operations, to be strictly spatially mapped to a spatially contiguous plot of AIE tiles and memtiles. The guaranteed spatial coexistence of all `air.herd` and L2 memory references ensure data flow between specialized AIE tiles via L2 memory.

The `air.segment` encapsulates the original computational logic, including L2 memory operations and the `air.herd` operation, which itself represents a computational kernel optimized for a matrix of AIE compute tiles. This encapsulation suggests that the computation can be offloaded in a virtualized AIE device cloud, as long as the resource requirements are fulfilled.

The transformation retains the explicit management of L2 memory (allocations and copies) and computations (the `air.herd` operation) but organizes them within the `air.launch` construct. This organization could facilitate the dynamic hardware dispatch managed from the host-side control program, scheduled by the compiler or runtime system, taking advantage of hardware capabilities for parallel execution.

`air-copy-to-dma`

Converts memory operations to optimize data transfer through Direct Memory Access (DMA) operations, targeting AIE's DMA hardware.

*Input IR:*
The input IR contains some generic memory copy operations (`memref.copy` or `linalg.copy`), moving data between L1, L2 or L3 memories.


*Output IR:*
The transformed code replaces some of the generic memory copy operations with `air.dma_memcpy_nd` operations. These DMA operations are specialized for direct memory-to-memory transfers, bypassing the host and thus reducing data transfer latency and host workload. The `air.dma_memcpy_nd` operation is designed to handle n-dimensional data transfers efficiently, explicitly representing the n-dimensional data rearrangement pattern on the fly.

Optimization of Data Transfer Dimensions: The transformation includes specifying the *offsets,* *sizes* and *strides* for the DMA operations explicitly, counted in number of elements (e.g. `[%c0_11, %arg12] [%c128_12, %c256_13] [%c1024_14, %c1_15]`).

Corresponding to the DMA operations, there are allocations (`memref.alloc()`) for temporary storage and deallocations (`memref.dealloc()`) for these temporary spaces after the DMA operations are complete. This ensures that the memory resources are managed efficiently and are only used when needed.

The core computational logic, represented by operations like `linalg.generic`, remains intact. The focus of the `air-copy-to-dma` pass is on optimizing the data movement that surrounds or supports these computations, rather than altering the computations themselves.

The DMA operations include identifiers (`{id = 1 : i32}, {id = 2 : i32}, {id = 3 : i32}`), which could be used for tracking, debugging, or further optimization by subsequent passes or by the runtime system. This explicit tagging could help in analyzing the performance or behavior of DMA operations within the execution pipeline.

`air-dependency`

Transforms the original IR towards an asynchronous execution model, specifically targeting optimizations and parallel execution of memory operations, computational tasks, and data transfers.

*Input IR:*
A generic MLIR IR representing some compute or data movement operations to be executed on hardware.

*Output IR:*
The pass introduces asynchronous air.execute operations that encapsulate both data transfer (`air.dma_memcpy_nd`) and computational (`linalg.generic`) tasks. These tasks are now explicitly managed through asynchronous tokens (`%async_token`), which represent dependencies between operations. The `air.wait_all` operation is used to synchronize on multiple asynchronous tokens, effectively waiting for all dependent operations to complete before proceeding. This allows for overlapping data transfers with computation, reducing idle time and improving overall execution efficiency.

By leveraging asynchronous tokens and explicit waits (`air.wait_all`), the pass enforces a clear execution order based on data dependencies. This explicit management helps in ensuring that operations are executed as soon as their prerequisites are met, potentially in parallel, leading to better resource utilization and shorter execution times.

If the input code was organized as nests and segments of `air.herd`, `air.segment` and `air.launch`, the transformed code shall also explicitly manage how they can be executed asynchronously (`air.launch async`). This segmentation, along with the assignment of unique identifiers to operations ({id = X : i32}), suggests a fine-grained control over the execution flow, enabling optimizations like concurrent execution of independent code segments.

The pass separates data preparation (e.g., memory allocations and data transfers) from the computational workload. By doing so, it allows for data transfers to be overlapped with computations, minimizing the overall execution latency. This is particularly beneficial to AIEs featuring discrete hardware DMA, which is faster and more efficient than host-driven memory transfers, as they can move data directly between memory locations without host intervention.

`air-dependency-schedule-opt`

Optimizes the scheduling of dependencies in the MLIR module, specifically focusing on improving the parallelism and efficiency of execution through the asynchronous execution model. One key optimization introduced is the detection of broadcasting data movement opportunities, and the subsequent inferring of broadcasting pattern.

*Input IR:*
The input IR has the L1 memory management and computation encapsulated within `air.herd`, and any direct memory-to-memory transfers represented with `air.dma_memcpy_nd` operations.

*Output IR:*
The transformation pass optimizes some `air.dma_memcpy_nd` operations within the body of `air.herd` by introducing a `broadcast_pattern` attribute. The presence of a `broadcast_pattern` attribute in an `air.dma_memcpy_nd` operation suggests that the data transfer is being optimized to replicate data across multiple destinations in a specific pattern. 

`air.herd` orchestrates parallel execution of computational tasks across multiple AIE tiles, and
specifying a broadcast_pattern for data transfers in its body reduces the bandwidth bottlenecks related to the finite AIE DMA channels.

The specific details of the broadcast_pattern attribute (e.g., `affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 3 >= 0, s0 >= 0, -s0 + 3 >= 0)>`) describe the rules that govern the data  replication pattern. Specifically, the symbol `s0` represents an iteration across unique broadcast sources, while the dimensions `d0` and `d1` represent the two-dimensional broadcast destination space: the set in `d0, d1` space corresponding to a `true` result in the `affine_set` represents all broadcast destinations to the broacast source `s0`. For example, in the `affine_set` given above, for each integer value of `s0`, ranged within `[0, 3]`, the set returns `true` if `(d0 == s0, 0 <= d1 <= 3`, meaning that this DMA operation shall involve four unique broadcast sources, where each source is broadcasted four-way across the 2nd (`d1`) dimension.

`dma-to-channel`

Transforms direct memory access (DMA) operations into channel-based communications, consisting of a series of channel put and get operations via shared channel constructs.

*Input IR:*
The input IR has the L1 memory management and computation encapsulated within `air.herd`, and any direct memory-to-memory transfers represented with `air.dma_memcpy_nd` operations. The input IR may or may not have MLIR-AIR asynchronous interface.

*Output IR:*
The transformation introduces various `air.channel` entities, each configured for specific communication patterns (e.g., sizes `[1, 1]` or `[4, 4]`, `broadcast_shapes`). Direct memory accesses (`air.dma_memcpy_nd`) are replaced by channel operations (`air.channel.put`, `air.channel.get`), explicitly representing data being sent or received from memory through channels, i.e. AIE tile-to-tile streaming interconnect.

The use of channels for communication also implies a synchronization mechanism. When data is put into or retrieved from a channel, the operations are inherently synchronized with the data's availability, introducing more deterministic execution patterns compared to direct memory access.

By organizing data movement through channels, the transformed IR is better suited for parallel execution across AIE tiles, where channels can facilitate the broadcasting of data across multiple AIE tiles.

`air-dependency-canonicalize`

Transforms the input IR by optimizing and restructuring the dependency and execution flow of the operations without altering the semantics of the original program. 

*Input IR:*
The pass consists of asynchronous data transfer, computational and synchronization tasks, explicitly managed through asynchronous tokens (`%async_token`), which represent dependencies between operations.

*Output IR:*
The transformation pass reduces the unnecessary propagation and handling of async tokens in the task graph, leading to cleaner and potentially more efficient code representation.

The pass also optimizes unnecessary asynchronous events, including memory allocation, deallocation and scalar arithmetic operations (`memref.alloc,` `memref.dealloc,` `arith` and `affine.appy`), by analyzing the lifetime and usage of their results and yielded async tokens, thus reducing overhead and improving memory usage efficiency.

The pass simplifies the loop-carred async tokens in control flow constructs (`scf.for`, `scf.parallel`) and potentially merges or eliminates redundant control flow paths. This makes the program easier to understand and can help in further optimization passes.

`air-split-l2-memref`

Transforms the input IR by splitting certain L2 memory references (`memrefs`) to adhere to AIE memtile-specific buffer and DMA channel constraints or optimization opportunities.

*Input IR:*
The input IR consists of some L2 memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations.

*Output IR:*
Large L2 `memrefs` are split into smaller chunks, if the `air.channel` data access pattern implies spatial parallelism. `air.channel` data access pattern may infer memref splitting opportunity if multiple data consumers or producers access chunks of the memory in non-overlapping pattern. This pattern is explicitly represented via `air.channel`, where the `memref` is accessed by multiple puts or gets via either multiple `air.channel` declarations, or multiple sub-channels of an `air.channel` declaration.

Once L2 memref splitting opportunities are detected, the pass then replaces them with smaller, more manageable pieces, and introduces new channel declarations (`@channel_X`) for each new memref, making downstream memtile buffer allocator easier to manage and optimize for bandwidth utilization.

The pass introduces explicit memory allocation and deallocation operations for the newly created L2 memrefs, thus explicitly managing their lifetime and facilitating their bufferization onto memtiles in the downstream pass.

`air-isolate-async-dma-loop-nest`

Transforms the IR by splitting the `scf.for` loop nests in `air.segment` body, based on the loop-carried async dependency of the original loop. The goal is to attempt to split nested `scf.for` loops around `air.channel.put/get` operations into perfect `scf.for` loop nests where only a single `air.channel.put`, or `air.channel.get` operation exists. This exposes DMA channel optimization opportunities, where perfectly nested parent for loops can fold as extra DMA channel wrap-and-stride dimensions.

*Input IR:*
The input IR consists of some L2 memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations under some imperfect `scf.for` loop nest. L2 memory operations have been encapsulated within `air.segment`.

*Output IR:*
The pass analyzes any imperfectly nested `scf.for` loop which contains `air.channel.put` and `air.channel.get` operations and detemine, based on loop-carried async dependency paths passing through the channel operations, whether the loop is splittable into new asynchronously parallel `scf.for` loop nests, which only contain a single channel operation in its innermost iteration space.

This pass exposes channel operation optimization opportunities, where any perfect for-loop nests, parent to a channel operation, can be folded as new wrap-and-stride dimensions in its n-dimensional data layout rearrangement pattern. The isolation of perfect for-loop nests, and their subsequent folding into channel operations, is especially useful for AIE architecture, which features memtiles possessing large memory space and high DMA streaming bandwidth, but no core to inference any control flow around its L2 memories.

In addition, the pass also eliminates any perfectly nested for loops surrounding any `air.herd`. The rationale is that the `air.herd` body will be outlined into compute tasks executed on AIE cores as well as finite-state machines of AIE tile DMA Buffer Descriptors, both of which are executing within infinite loops.

`air-loop-fusion`

Optimizes the data movement around L2 memories by rearranging and potentially fusing perfect `scf.for` loop nests of `air.channel.put` and `air.channel.get`, which access the same L2 memref, into `scf.for` loop nest patterns mappable to a complex finite-state machine consisting of a multiple of AIE DMA Block Descriptors.


*Input IR:*
The input IR consists of some L2 memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations, nested under some perfect `scf.for` loop nest. L2 memory operations have been encapsulated within `air.segment`.

*Output IR:*
The pass attempts to merge perfectly nested for loops, which iterate over `air.channel.put` or `air.channel.get` accessing the same L2 `memref`, into a single loop. This can greatly facilitate any downstream conversion to physical representations of memory operations around the `memref`, using only a sequence of low-level constructs such as Block Descriptors.

The pass also attempts to confine the lifetime of the target L2 `memref` by moving its allocation `memref.alloc()` and deallocation `memref.dealloc()` into the begin and end of the fused for loop body, in order to facilitate the downstream compilation.

Within the confined lifetime of the L2 `memref`, the pass also attempts to analyze its effective data access pattern at each time phase in its lifetime, to determine whether its size can be shrunk. If true, then the memref's allocation and deallocation is updated to a reduced size, to optimize for memory utilization.

After loop fusion, the compilation pass analyzes the data production and consuption pattern around the L2 memref, and reconstructs the loop-carried asynch dependency within the fused loop body. The newly constructed dependency edges represent the transition edges connecting the states (i.e. `air.channel.put` and `air.channel.get` operations) of the finite-state machine.

`air-label-scf-for-to-ping-pong`

The analysis pass detects ping-pong buffering opportunities implied in a `scf.for` loop, based on inspection of the data production and consumption around a `memref` with a lifetime within its body. Upon successful detection of ping-pong buffering opportunities, the pass assigns an `unroll=2` attribute, as compiler flag for a downstream pass to transform the loop into explicitly representing the ping-pong schedule.

*Input IR:*
The input IR consists of some memory references (`memrefs`) accessed by `air.channel.put/get` data movement operations and `linalg.generic` compute operations, nested under some `scf.for` loop. The `memref` allocation and deallocation are located within the for loop iteration space, if its lifetime is strictly scoped within each iteration of the for loop. The `scf.for` loop needs to have MLIR-AIR async interface, in order for any subsequent ping-pong transformation (`air-ping-pong-transform`) pass to successfully represent the ping-pong schedule via async tokens.

*Output IR:*
The pass analyzes the computational graph of an `scf.for` loop body, and detect whether its hardware schedule is ping-pong transformable. Currently, any `scf.for` loop with any `memref` whose lifetime is scoped within its iteration, is considered as ping-pong transformable.

`air-ping-pong-transform`

Transforms the IR into explicitly representing a ping-pong buffered hardware schedule, via unrolling the `scf.for` loop by two, and explicitly representing a multiple of async threads passing through the `scf.for` loop body in parallel.

*Input IR:*
The input IR consists of some memory references (`memrefs`) accessed by `air.channel.put` and `air.channel.get` data movement operations, nested under some `scf.for` loop, possibly labelled with an attribute `unroll=2` which is generated by an upstream pass (`air-label-scf-for-to-ping-pong`).


*Output IR:*
The pass transforms the computational graph of an `scf.for` loop body---previously detected by `air-label-scf-for-to-ping-pong` as being ping-pong transformable---to use "ping-pong" buffers and channels for asynchronous execution and data transfer, aiming to reduce memory access latency and increase throughput by overlapping computation with data transfer and utilizing local memory more effectively.

The pass is first unrolled by a factor of two to generate explicit "ping" and "pong" `memrefs` and async events, as handles to explicitly represent the parallelism between "ping" data producers and "pong" data consumers, and vice versa.

The `scf.for` body is transformed into having a multiple of async tokens (`%async_token`) passed into its body via `iter_args` operands/arguments, and yielded and returned via `scf.yield`. Each of those tokens represents an async thread flowing through a path made of async dependency edges across the for-loop iterations.

`air-collapse-herd`

Transforms the shape of `air.herd` by attempting to collapse to occupy complete columns on AIE device.

*Input IR:*
The input IR has the L1 memory management and computation encapsulated within `air.herd`.

*Output IR:*
The pass attempts to collapse the `air.herd` to the left, attempting to occupy complete columns of AIE tiles. The attempt will stop if the number of tiles in `air.herd` exceeds the user provided `max-col-size` option.

`air-place-herds`

Places `air.herd` within its parent `air.segment` using Greedy method; infers `air.segment` shape based on the shape and size of all `air.herd` operations within its body.

*Input IR:*
The input IR has the L1 memory management and computation encapsulated within `air.herd`, and L2 memory management and `air.herd` encapsulated within `air.segment`.

*Output IR:*
After the pass has been applied, both the `air.segment` and `air.herd` operations have additional attributes (`x_loc, x_size, y_loc, y_size` for the segment, and `x_loc, y_loc` for the herd). These new attributes specify explicit spatial placement and size information:

`x_loc` and `y_loc` denote the starting location of the segment or herd in a two-dimensional space (a grid of AIE tiles).
`x_size` and `y_size` specify the dimensions of the segment, indicating how much space the segment occupies in each dimension.
The pass is responsible for spatially placing `air.herd` opertaions onto a grid of AIE tiles using Greedy method.

`air-to-aie`

Convertss an input IR from MLIR-AIR dialect into MLIR-AIE dialect, for efficient mapping onto AIEs.

*Input IR:*
The input MLIR-AIR dialect code specifies the memory allocation and management, execution synchronization, and the computation of a hardware schedule. It uses constructs such as `air.execute`, `air.herd`, `air.channel.put/get`, and `memref` operations to describe a hardware schedule of computation and data movement in a platform-agnostic manner. This includes managing data movement, synchronization tokens for asynchronous execution, and detailed sub-tasks like broadcasting data copy and ping-pong buffering for optimal access patterns.

*Output IR:*
The output MLIR-AIE dialect code is much more hardware-specific. It maps the abstract operations into concrete actions on the AIE tiles, including data movement between tiles (using DMA operations), buffer allocation on specific tiles, and computation instructions for the AI Engines.

The input IR is cloned, where the original copy lowers to MLIR-AIE dialect and the cloned copy gets lowered to runtime program by a downstream pass (`air-to-std`). The SHIM DMA copy operations in the runtime program maintains linkage to the MLIR-AIE accelerator's physical SHIM DMA channel via metadata annotations (e.g., `metadata = @airMemcpyIdXX`).

Lowered from `air.channel.put` and `air.channel.get`, DMA operations (`aie.dma_start`, `aie.dma_bd`) are explicitly detailed, showing how data is moved between buffer and processing cores. This is crucial for performance, as efficient data movement is key to achieving high throughput on the AIE architecture.

The combination of multiple data consumers and producers (e.g. `air.channel.put/get` and `linalg.generic`), connected via a chain of `scf.for` loop-carried async tokens, form a representation of finite-state machines, which are subsequently lowered into a combination of lock (`aie.lock`) acquire-release actions, DMA Block Descriptors (`aie.dma_bd`) and AIE core compute kernels. Locks ensure that data dependencies are respected, and resources are not accessed simultaneously by multiple tiles in a way that could lead to conflicts or data corruption.

The computation (`linalg.generic`, `func.call` or `arith`/`scf` loop nests of `arith` scalar operations) is divided across multiple AIE cores, with specific L1 buffers (`aie.buffer`) and computational kernels outlined to each core. This shows how the program is parallelized and distributed across the hardware.

The AIE core code includes explicit control flow for data movement and computation (`cf.br`, `scf.for` loops) iterating over data chunks.

`air-to-std`

Converts the MLIR-AIR dialect code into AIRRt dialect which represents the runtime code dispatching the program by pushing and pulling data from the AIE SHIM tile DMAs.

*Input IR:*
High-Level to Low-Level: The original code operates at a higher level of abstraction, specifying operations like compute, data transfer, and synchronization primitives.

*Output IR:*
After the conversion, the control flow, synchronization and DMA operations responsible for external memory (L3) copies via AIE SHIM DMAs are retained, while other operations are discarded. They are replaced with the AIRRt dialect operations which are dedicated to the representation of runtime operations, including explicit memory allocation (`airrt.alloc`), SHIM DMA data movement (`airrt.dma_memcpy_nd`), and synchronization (`airrt.wait_all`).

The use of `airrt.dma_memcpy_nd` after transformation shows detailed control over data movement, specifying the n-dimensional offsets, wraps and strides of data layout to be moved through SHIM DMA. This level of control is essential for optimizing data transfers between the AIE accelerator and external memory, where bandwidth is a limiting factor.

The control flow is represented with `affine` and `scf` loops, either retained from the input IR, or lowered from the `air.launch` iteration space.

The transformation includes explicit synchronization points (`airrt.wait_all`), which are crucial for managing dependencies between parallel operations, ensuring correct execution order without unnecessary stalls.

`affine-loop-opt`

Transforms the IR by optimizing `affine` loop nests. `affine` loop nests are loops that iterate over multi-dimensional arrays in a manner that can be described using `affine` transformations.

*Input IR:*
The input AIRRt dialect code represents the runtime program for the AIE accelerator, where the control flow is represented as `affine` and `scf` loops.

*Output IR:*
The most notable transformation applied by the pass is loop tiling on `affine` loop nests containing `airrt.dma_memcpy_nd` operations in innermost loop. The pass attempts to tile the loop nest by user-provided factors (option `affine-opt-tile-sizes`). The tiled loops are then unrolled by a downstream pass (`air-unroll-outer-affine-loops`) to give an unrolled sequence of SHIM DMA Block Descriptors.

`air-unroll-outer-affine-loops`
Unrolls the two outermost dimensions in `affine` loop nests of the AIRRt runtime code.

*Input IR:*
The input IR contains nested `affine.for` loops, previously tiled by an upstream pass (`affine-loop-opt`) into having a desirable number of dimensions. Inside the loops, there are operations like `airrt.segment_load`, `arith.constant`, `arith.index_cast`, `airrt.dma_memcpy_nd`, and `airrt.wait_all`, indicating memory operations, arithmetic operations, and synchronization points. These operations are related to loading the binaries for AIE tiles, performing computation, and ensuring data is moved efficiently between the AIE device and external memory.

*Output IR:*

`airrt-to-ipu`
Converts the runtime program, described in AIRRt dialect, into instruction sequence specific to the LX6 controllers on Ryzen AI platform.

*Input IR:*

*Output IR:*
The output is significantly simplified and optimized compared to the input. It focuses on the data movement instructions on LX6 (`aiex.ipu.dma_memcpy_nd`) driving the data movement between the AIE accelerator and external memory via the SHIM DMA.

The transformed code introduces `aiex.ipu.sync` operations used to reprogram all DMA Block Descriptors in a SHIM DMA.

The function signature in the output code (e.g. `func.func @matmul_512x512_1024xi32__dispatch_0_matmul_512x512x1024_i32(%arg0: memref<512x1024xi32>, %arg1: memref<1024x512xi32>, %arg2: memref<512x512xi32>)`) takes the L3 `memrefs` to external memory as input.
