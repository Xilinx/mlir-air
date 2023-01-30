# AIR Asynchronous Concurrency

A CDFG abstraction is adopted in MLIR-AIR to represent asynchronous concurrency in an MLIR program.
This abstraction contains compiler passes which progressively extracts, analyses and optimizes the program's CDFG, in order to model asynchronous executions in AI Engine.

## Compilation pipeline

### Step one: extract CDFG from the MLIR-AIR program using `-air-dependency`

Below is a code snippet showing the memory allocation, data movement and computation of a tiled matrix multiplication program in MLIR-AIR, in _synchronous_ execution order.
```
scf.for %arg11 = %c0 to %c128 step %c32 {
  %5 = memref.alloc() : memref<16x32xf32, 2>
  %6 = memref.alloc() : memref<32x64xf32, 2>
  %7 = memref.alloc() : memref<16x64xf32, 2>
  air.dma_memcpy_nd (%5[] [] [], %arg8[%3, %arg11] [%c16, %c32] [%c128, %c1_0]) {id = 1 : i32} : (memref<16x32xf32, 2>, memref<128x128xf32>)
  air.dma_memcpy_nd (%6[] [] [], %arg9[%arg11, %4] [%c32, %c64] [%c128, %c1_0]) {id = 2 : i32} : (memref<32x64xf32, 2>, memref<128x128xf32>)
  air.dma_memcpy_nd (%7[] [] [], %arg10[%3, %4] [%c16, %c64] [%c128, %c1_0]) {id = 3 : i32} : (memref<16x64xf32, 2>, memref<128x128xf32>)
  linalg.matmul ins(%5, %6 : memref<16x32xf32, 2>, memref<32x64xf32, 2>) outs(%7 : memref<16x64xf32, 2>)
  air.dma_memcpy_nd (%arg10[%3, %4] [%c16, %c64] [%c128, %c1_0], %7[] [] []) {id = 4 : i32} : (memref<128x128xf32>, memref<16x64xf32, 2>)
  memref.dealloc %5 : memref<16x32xf32, 2>
  memref.dealloc %6 : memref<32x64xf32, 2>
  memref.dealloc %7 : memref<16x64xf32, 2>
}
```

`-air-dependency` pass automatically analyzes the data dependency and loop-carried dependency in code, generates a CDFG object in the compiler backend, and updates the MLIR-AIR program with AIR Async operation interface.
The post-analysis _asynchronous_ MLIR-AIR program is shown below.

```      
%2 = air.wait_all async [%async_token_8, %async_token_10]  {id = 2 : i32}
%3 = scf.for %arg11 = %c0 to %c128 step %c32 iter_args(%arg12 = %2) -> (!air.async.token) {
  %async_token_12, %results_13 = air.execute -> (memref<16x32xf32, 2>) {
    %alloc = memref.alloc() : memref<16x32xf32, 2>
    air.execute_terminator %alloc : memref<16x32xf32, 2>
  } {id = 8 : i32}
  %async_token_14, %results_15 = air.execute -> (memref<32x64xf32, 2>) {
    %alloc = memref.alloc() : memref<32x64xf32, 2>
    air.execute_terminator %alloc : memref<32x64xf32, 2>
  } {id = 9 : i32}
  %async_token_16, %results_17 = air.execute -> (memref<16x64xf32, 2>) {
    %alloc = memref.alloc() : memref<16x64xf32, 2>
    air.execute_terminator %alloc : memref<16x64xf32, 2>
  } {id = 10 : i32}
  %4 = air.dma_memcpy_nd async [%async_token_12, %arg12] (%results_13[] [] [], %arg8[%results_9, %arg11] [%c16, %c32] [%c128, %c1_7]) {id = 1 : i32} : (memref<16x32xf32, 2>, memref<128x128xf32>)
  %5 = air.dma_memcpy_nd async [%async_token_14, %arg12] (%results_15[] [] [], %arg9[%arg11, %results_11] [%c32, %c64] [%c128, %c1_7]) {id = 2 : i32} : (memref<32x64xf32, 2>, memref<128x128xf32>)
  %6 = air.dma_memcpy_nd async [%async_token_16, %arg12, %arg12] (%results_17[] [] [], %arg10[%results_9, %results_11] [%c16, %c64] [%c128, %c1_7]) {id = 3 : i32} : (memref<16x64xf32, 2>, memref<128x128xf32>)
  %async_token_18 = air.execute [%arg12, %5, %6, %4] {
    linalg.matmul ins(%results_13, %results_15 : memref<16x32xf32, 2>, memref<32x64xf32, 2>) outs(%results_17 : memref<16x64xf32, 2>)
  } {id = 11 : i32}
  %7 = air.dma_memcpy_nd async [%arg12, %async_token_18] (%arg10[%results_9, %results_11] [%c16, %c64] [%c128, %c1_7], %results_17[] [] []) {id = 4 : i32} : (memref<128x128xf32>, memref<16x64xf32, 2>)
  %async_token_19 = air.execute [%async_token_18] {
    memref.dealloc %results_13 : memref<16x32xf32, 2>
  } {id = 12 : i32}
  %async_token_20 = air.execute [%async_token_18] {
    memref.dealloc %results_15 : memref<32x64xf32, 2>
  } {id = 13 : i32}
  %async_token_21 = air.execute [%7] {
    memref.dealloc %results_17 : memref<16x64xf32, 2>
  } {id = 14 : i32}
  %8 = air.wait_all async [%arg12, %7]  {id = 1 : i32}
  scf.yield %8 : !air.async.token
}
```

### Step two: CDFG canonicalization using `-canonicalize` and `-air-dependency-canonicalize`

MLIR-AIR provides direct canonicalization pass `-canonicalize` to transform the MLIR-AIR async operations to canoncal form.

Many CDFG edges, represented as async dependency tokens in an operation's dependence list, are redundant as they do not represent the dominant producer-consumer relationships.
MLIR-AIR provides a graph canonicalization pass `-air-dependency-canonicalize` which removes redundant edges in the CDFG using _transitive reduction_ algorithm.

### Optional: visualize CDFG using `-air-dependency-parse-graph`

The CDFG which represents the MLIR-AIR program's asynchronous concurrency can be visualized as a `.dot` file using `-air-dependency-parse-graph` pass.
The generated dot file can be rendered using Graphviz.
The rendered CDFG below is generated from the matrix multiplication example in previous section.

<img src="assets/images/air_cdfg_example.svg">

### Step three: CDFG optimization using `-air-dependency-schedule-opt`

Having extracted a CDFG from the MLIR-AIR code, the `-air-dependency-schedule-opt` pass performs scheduling optimization by detecting and transforming inefficient code patterns in CDFG.
Data broadcasting opportunities across spatially mapped AIE cores are automatically detected and labelled.
Below is the output code snippet lowered from the tiled matrix multiplication example above.
Here, data broadcasting opportunities are labelled with keyword `broadcast_pattern`, followed by an affine set representing the address pattern of broadcast copy recepients.

```
#set = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 7 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 7 >= 0)>
%2 = air.wait_all async [%async_token_10, %async_token_8] 
%3 = scf.for %arg11 = %c0 to %c128 step %c32 iter_args(%arg12 = %2) -> (!air.async.token) {
  %async_token_12, %results_13 = air.execute -> (memref<16x32xf32, 2>) {
    %alloc = memref.alloc() : memref<16x32xf32, 2>
    air.execute_terminator %alloc : memref<16x32xf32, 2>
  }
  %async_token_14, %results_15 = air.execute -> (memref<32x64xf32, 2>) {
    %alloc = memref.alloc() : memref<32x64xf32, 2>
    air.execute_terminator %alloc : memref<32x64xf32, 2>
  }
  %async_token_16, %results_17 = air.execute -> (memref<16x64xf32, 2>) {
    %alloc = memref.alloc() : memref<16x64xf32, 2>
    air.execute_terminator %alloc : memref<16x64xf32, 2>
  }
  %4 = air.dma_memcpy_nd async [%async_token_12, %arg12] (%results_13[] [] [], %arg8[%results_9, %arg11] [%c16, %c32] [%c128, %c1_7]) {broadcast_pattern = #set, id = 1 : i32} : (memref<16x32xf32, 2>, memref<128x128xf32>)
  %5 = air.dma_memcpy_nd async [%async_token_14, %arg12] (%results_15[] [] [], %arg9[%arg11, %results_11] [%c32, %c64] [%c128, %c1_7]) {broadcast_pattern = #set1, id = 2 : i32} : (memref<32x64xf32, 2>, memref<128x128xf32>)
  %6 = air.dma_memcpy_nd async [%async_token_16, %arg12] (%results_17[] [] [], %arg10[%results_9, %results_11] [%c16, %c64] [%c128, %c1_7]) {id = 3 : i32} : (memref<16x64xf32, 2>, memref<128x128xf32>)
  %async_token_18 = air.execute [%6, %5, %4] {
    linalg.matmul ins(%results_13, %results_15 : memref<16x32xf32, 2>, memref<32x64xf32, 2>) outs(%results_17 : memref<16x64xf32, 2>)
  }
  %7 = air.dma_memcpy_nd async [%async_token_18] (%arg10[%results_9, %results_11] [%c16, %c64] [%c128, %c1_7], %results_17[] [] []) {id = 4 : i32} : (memref<128x128xf32>, memref<16x64xf32, 2>)
  %async_token_19 = air.execute [%async_token_18] {
    memref.dealloc %results_13 : memref<16x32xf32, 2>
  }
  %async_token_20 = air.execute [%async_token_18] {
    memref.dealloc %results_15 : memref<32x64xf32, 2>
  }
  %async_token_21 = air.execute [%7] {
    memref.dealloc %results_17 : memref<16x64xf32, 2>
  }
  scf.yield %7 : !air.async.token
}
```
