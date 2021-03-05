# AIR add one example

This example shows an end to end flow with very little extra user code required.  The code adds 1 to a vector of integers based on this PyTorch function

```python
with builder.capture_function("graph", [t0]) as f:
    t2 = t0 + torch.tensor(1, dtype=torch.int32)
    f.returns([t2])
```

## First passes: Export to ATen dialect

The first passes to take the Python code and turn it into the ATen dialect are now made available to us thanks to the good folks at npcomp.  By just running the Python code in this example, we get our first MLIR version of the application using the ATen dialect

```llvm
module  {
  func @graph(%arg0: tensor<256xi32>) -> tensor<256xi32> {
    %cst = constant dense<1> : tensor<i32>
    %c1_i64 = constant 1 : i64
    %0 = "aten.add"(%arg0, %cst, %c1_i64) : (tensor<256xi32>, tensor<i32>, i64) -> tensor<256xi32>
    return %0 : tensor<256xi32>
  }
}
```

## Second passes: Turn into affine loops

The second set of passes take this initial MLIR, pattern match on the functions in it, and then turn the whole thing into an affine loop nest. In this case, we can clearly see the loop bounds need to be 256 to go over the whole 256 element output tensor.

```llvm
    affine.for %arg1 = 0 to 256 {
      %3 = affine.load %1[%arg1] : memref<256xi32>
      %c1_i32 = constant 1 : i32
      %4 = addi %3, %c1_i32 : i32
      affine.store %4, %0[%arg1] : memref<256xi32>
    } {affine_opt_label = "affine_opt"}
```
The loop is also labelled ready for the next set of transforms, which will optimize this affine loop based on user parameters

## Third passes: First level of tiling, and first data movement

The first level of tiling breaks the problem into compute unit sized tiles.  In this case, the user says they would like these tiles to each contain 64 elements of the overall tensor.  The user also says that the overall tensor needs to get copied into the memory level 2 - the id for the memories next to the compute units.  The default location of tensors is memory space 0, which for us is external memory.
This results in a 2 deep loop nest: an outer loop which goes over all the tiles, and the inner loop which is going over the individual elements in the tiles of the tensors.  DMA operations are also generated to indicate where moves are required.  At this stage, this is all still abstract.  We haven't said what is going to be done in time or space.  Just that there's a sequence of operations.

```llvm
    affine.for %arg1 = 0 to 256 step 64 {
      %3 = alloc() : memref<64xi32, 2>
      %4 = alloc() : memref<1xi32>
      affine.dma_start %1[%arg1], %3[%c0], %4[%c0], %c64 : memref<256xi32>, memref<64xi32, 2>, memref<1xi32>
      affine.dma_wait %4[%c0], %c64 : memref<1xi32>
      %5 = alloc() : memref<64xi32, 2>
      %6 = alloc() : memref<1xi32>
      affine.for %arg2 = #map0(%arg1) to #map1(%arg1) {
        %7 = affine.load %3[-%arg1 + %arg2] : memref<64xi32, 2>
        %c1_i32 = constant 1 : i32
        %8 = addi %7, %c1_i32 : i32
        affine.store %8, %5[-%arg1 + %arg2] : memref<64xi32, 2>
      }
      affine.dma_start %5[%c0], %0[%arg1], %6[%c0], %c64 : memref<64xi32, 2>, memref<256xi32>, memref<1xi32>
      affine.dma_wait %6[%c0], %c64 : memref<1xi32>
      dealloc %6 : memref<1xi32>
      dealloc %5 : memref<64xi32, 2>
      dealloc %4 : memref<1xi32>
      dealloc %3 : memref<64xi32, 2>
    } {affine_opt_label = "affine_opt"}
```

### Fourth passes: Assign tiles to cores

The second level of tiling is a simple transform to take the 4 steps of the outermost loop, and make a 2 deep loop nest, each with 2 steps.  This 2x2 then represents the 2x2 spatial grid of compute units we want to use to do the actual herd computation

```llvm
    affine.for %arg1 = 0 to 2 {
      affine.for %arg2 = 0 to 2 {
        ...
        affine.dma_start %1[%3], %4[%c0], %5[%c0], %c64 : memref<256xi32>, memref<64xi32, 2>, memref<1xi32>
        affine.dma_wait %5[%c0], %c64 : memref<1xi32>
        ...
        affine.for %arg3 = 0 to 64 {
          ...
        }
        affine.dma_start %6[%c0], %0[%3], %7[%c0], %c64 : memref<64xi32, 2>, memref<256xi32>, memref<1xi32>
        affine.dma_wait %7[%c0], %c64 : memref<1xi32>
        ...
}
```


### Fifth passes: Lower to the AIR herd

So far, this is all abstract. We have talked about DMAs and we have inner loops but now we want to lower this onto the virtual AIR platform.  The outer loop are gathered into a single "launch_herd" construct to contain the launching of all the cores, and their data movement

```llvm
    air.launch_herd tile (%arg1, %arg2) in (%arg3=%c2, %arg4=%c2) args(%arg5=%1, %arg6=%0) : memref<256xi32>,memref<256xi32> {
      ...
      air.dma_memcpy (%6, %arg5, [%c0], [%5], %c64) {id = 1 : i32} : (memref<64xi32, 2>, memref<256xi32>, [index], [index], index) -> ()
      ...
      affine.for %arg7 = 0 to 64 {
        ...
      }
      air.dma_memcpy (%arg6, %7, [%5], [%c0], %c64) {id = 2 : i32} : (memref<256xi32>, memref<64xi32, 2>, [index], [index], index) -> ()
      ...
      air.herd_terminator
    }
```

### Sixth passes: Lower to the ACAP implementation of AIR, and build C functions for the runtime

These passes now first build the logical ACAP dialect version of the AIR implementation.  This file contains all the information required to make a static herd implementation on the AI Engine array, and so by passing through the usual logical->physical->C function flow, we can end up with C functions which the runtime can call to configure the AIE array with the herd.

### Seventh passes: Build the elfs for the four cores

The logical dialect view of the herd can also be used to generate the actual elf files used by each core.  Peano ends up getting called four times with this body to produce the elfs for the 2x2 array of AI Engines.  The runtime can then load these into the appropriate core by name matching

### Eight passes: Build a .so for the runtime

Since the loop information containing DMAs and compute is still avaiable, these passes transform this into an executable form for the runtime on the ARM.  The basic idea is that the whole application is made available as a run function in a .so, and will call out to the runtime for concrete implementations of data movement.  At the current time, this is a little constrained, since the sixth passes have locked down all the routing, which means the DMAs to be used is locked down as is the herd placement.  Future releases will relax this constraint, and allow the runtime to truly make dynamic decisions about placement, and do any routing fixup required.

At the current time, it is the responsibility of the user to write the concrete data movement function.  This function is passed a unique transfer id (e.g. Arg A for the compute core), and the X/Y position in the herd of the core requesting the data movement.  We currently key off these triplets to figure out which shim DMA to program.  As noted above, this is a constrained problem since the physcial shim DMA allocation has been done at compile time.

Finally, in this example, a bounce buffer is currently used as the source or destination for the data movement, so there is an extra copy step.  A future version will use a specialty allocator to allow the runtime to get the PA of the tensor directly, and thus program the shim DMAs correctly.

## Running the test application

The directory contains a simple test program, with a concrete data move implementation using the ARM to set up the shim DMAs.  The test program also shows how to load the .so and call the graph.run function.
