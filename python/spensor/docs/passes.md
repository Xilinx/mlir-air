# Spensor Lowering Passes

This file is intended to introduce the deisgn and implementation lowering passes under spensor/passes

# Memory Analysis

# Parallel Analysis

# Expand Parallel

`ExpandParallel` rewrites operations on NDSpensor types into Spensor types wrapped by parallel loops. The bounds of 
parallel loops are decided in previous `ParallelAnalysis`. Patterns in this pass fall into three categories:
1. Parallel Entrance

   These patterns introduce parallel loops and replaces subsequent usages with subviews. For example:
    ```
    split_arg0 = split(arg0, num_partition = 4, dim = 0): <4xSpensor>
    add(split_arg0, split_arg0): <4xSpensor>
      |
      |
      V
    scf.parallel i: 0 -> 4
      split_arg0 = subview(arg0, i): Spensor
      add(split_arg0): Spensor
    ```
    Here, the split operation creates a parallel loop, and later uses of the operand are replaced by its subviews.
    Patterns in this category:
   + Split Pattern
   + SplitAll Pattern
   + A special case of Move Pattern (We are under L2, and it's a movement from L2 to L1. It needs to create a inner
   loop with `memory_tag` L1)
2. Regular Operations

    Patterns in this category simply rewrites operations from NDSpesnor to Spensor type, because their operands are
   supposed to be subviews or result from other Spensor operations. They include:
    + Add Pattern
    + Matmul Pattern
    + Regular Move Pattern
    + MoveTo Pattern
    + ReduceSum Pattern

   Because operations are polymorphic on both NDSpensor and Spensor types, operation names don't change here but only
types.

3. Operations with extra handling

    Operations in this category needs extra handling rather simply rewriting.
    + NDCombine Pattern
   
       `NDCombine` operation is special here because it actually allocates a new Spensor.
   
      Before lowering:
       ```
       scf.parallel i: 0 -> 4
          split_arg0 = subview(arg0, i): Spensor<T, L1>
          add_res = add(split_arg0): Spensor<T, L1>
      # Note, ndcombine hasn't been lowered 
      # so it's still on NDSpensor type
      comb_res = ndcombine(add_res, nd_dim=1, dim=1): NDSpensor<1xSpensor<4xT, L2>>
      move(comb_res, "L1")
      ```
      After lowering:
      ```
      comb_res = spensor.alloc_spensor: Spensor<4xT, L2>
      scf.parallel i: 0 -> 4
          split_arg0 = subview(arg0, i): Spensor<T, L1>
          add_res = add(split_arg0): Spensor<T, L1>
          ith_buffer = subview(buffer, i): Spensor<T, L1>
          moveTo(add_res, ith_buffer)
      move(comb_res, "L1")
      ```
      It creates a buffer sized as a multiple of the operand and rewrites usage accordingly.
   + NDReduce Pattern
   
      While `NDReduce` is slight different. It first combines the previous result, allocates another buffer, and appends
      a new parallel loop for accumulation.
   
      Before lowering:
       ```
       scf.parallel i: 0 -> 4
          split_arg0 = subview(arg0, i): Spensor<T, L1>
          add_res = add(split_arg0): Spensor<T, L1>
      # Note, ndreduce hasn't been lowered 
      # so it's still on NDSpensor type
      red_res = ndreduce(add_res): NDSpensor<1xSpensor<T, L2>>
      move(red_res, "L1")
      ```
      After lowering:
      ```
      buffer = spensor.alloc_spensor: Spensor<4xT, L2>
      red_res = spensor.alloc_spensor: Spensor<T, L2>
      scf.parallel i: 0 -> 4
          split_arg0 = subview(arg0, i): Spensor<T, L1>
          add_res = add(split_arg0): Spensor<T, L1>
          ith_buffer = subview(buffer, i): Spensor<T, L1>
          moveTo(add_res, ith_buffer)
      fill(red_res, 0)
      scf.parallel _: 0 -> 1
          scf.for i: 0 -> 4
              ith_buffer = subview(buffer, i): Spensor<T, L1>
              red_res = add(red_res, ith_buffer)
              moveTo(red_res, ith_buffer)
      move(red_res, "L1")
      ```
      Note: We implement in this way because it doesn't use the cascading feature recently merged. We should 
      switch to that way and code generation here could be easier.
    + NDRepeat Pattern
   
      `NDRepeat` repeats the operand, so we do nothing here but just replace the usage with its operand.
  

# Tiling Parallel

`TilingParallel` tiles parallel loops according to a specified memory shape.
Each parallel loop is expected to carry a `memory_tag` that indicates which memory level (e.g., L1, L2) the loop 
operates on. If a memory shape is provided in `declareMemoryOp`, this pass tiles the loop based on that shape.

This pass is necessary because when lowering to AIR, the upper bounds of the innermost loop are directly mapped to herd size.
If the loop bounds do not align with available hardware configurations, AIR reports an `Invalid Placement` error.

There are two open issues that need further discussion:
1. Tiling bounds (TODO)

      The goal is to maximize the number of active cores.
    For example, Phoenix has [4x4] L1 cores and we want to tile [8x8], the loop can be split into a for-loop with bounds [2x2] and herd size [4x4].
    Similarly, tiling [8] can map to herd size [4x2].
    This suggests a greedy algorithm where each L1 dimension is matched with one upper bound in the source loop.


2. Spatial/Temporal Tiling

    There are two kinds of tiling strategies here:
    ```
    scf.parallel (i, j): (0, 0) -> (8, 8)
      load buffer[i,j]
       |
       |
       V
    scf.for (i1, j1): (0, 0) -> (2, 2)
     scf.parallel (i2, j2): (0, 0) -> (4, 4)
       i = i1*4 + i2
       j = j1*4 + j2
       load buffer[i,j]
       ...
       
    -------------------------
    
    scf.parallel (i2, j2): (0, 0) -> (4, 4)
     scf.for (i1, j1): (0, 0) -> (2, 2)
       i = i1*4 + i2
       j = j1*4 + j2
       load buffer[i,j]
       ...
    
    ```
    
    In the first approach, each core loads one element from the buffer.
    In the second approach, each core loads a [2x2] elements.
    At present, it is unclear which is preferable. This should be made configurable during lowering.
     

# Spensor to Tensor

`SpensorToTensor` transfers operations on Spensor types to tensor types. More specifically, it includes following
patterns.
1. Function Transformation: func(arg0:Spensor, arg1:Spesnor) -> func(arg0:Tensor, arg1:Tensor)
2. Subview Pattern: spensor.subview -> tensor.extract_slice
3. Move Pattern: spensor.move -> bufferization.alloc_tensor + linalg.copy
4. MoveTo Pattern: spensor.move_to -> linalg.copy
5. Add Pattern: spensor.add -> linalg.add
6. Matmul Pattern: spensor.matmul -> linalg.matmul
7. Fill Pattern: spensor.fill -> linalg.fill
8. ReduceSum Pattern: spensor.reduce_sum -> scf.for within memref.load, memref.store, arith.add
9. AllocSpensor Pattern: spensor.alloc_spensor -> bufferization.alloc_tensor

# Append Constant

`AppendConstant` is a helper class for managing constant indices created with `arith.constantOp`.
In many cases, such as generating `scf.parallel` or `affine.apply`, constants are needed as loop bounds or offsets.
Without this helper, each pattern would manually construct constants, often duplicating the same values across the IR,
leading to unnecessary clutter. This pass centralizes constant creation and reduces visual noise.

When a pass needs an index constant, it calls `getConstantOpByIndex` in `spensor_util.py`.
This function checks whether the constant already exists in `index_to_constant_op` (defined in `spensor_global.py`).
If not, it creates the operation and records it. This ensures that identical constants always reference the same operation.

At the end of lowering, this pass must be run to insert all collected constants from `index_to_constant_op`
at the beginning of the function. Otherwise, constants would be emitted inline where they are used.
# Spensor Dead Code Elimination

`SpensorDeadCodeElimination` cleans up unused operations left after lowering.
For example, `spensor.declareMemoryOp` is only relevant to `MemoryAnalysis` and can be safely removed afterward.
The pass erases an operation only if nothing else depends on it.
