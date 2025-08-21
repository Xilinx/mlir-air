# Spensor Lowering Passes

This file is intended to introduce the deisgn and implementation lowering passes under spensor/passes

# Memory Analysis

# Parallel Analysis

# Expand Parallel

# Tiling Parallel

`TilingParallel` tries to tile a parallel loop by specified memory shape. It's expected every parallel loop
is attached with a `memory_tag` stating the loop is operated at which memory level such as L1 or L2. If a
memory shape is specified in `declareMemoryOp`, this pass tiles the loop by the shape. We need this pass because when
lowering to AIR, the upper bounds of the inner most loop are directly transferred to herd size, while it might not match
any hardware configurations and AIR reports an `Invalid Placement` error.
There are two remaining problems to need be discussed:

1. Tiling bounds (TODO)

   The basic idea here is to use as many cores as possible at once. For example, Phoenix has [4x4] number of L1 cores,
we want to tiling [8x8] into a for loop with bounds [2x2] and the herd size is [4x4]. In addition, [8] into a herd size 
[4x2]. This should be a greedy algorithm that each L1 dimension should be mapped to one upper bound in source loop,


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
In the first tiling strategy, each cores still loads one data from buffer.
However in the second one, each core loads [2x2] data from buffer. I don't have an answer
for now to justify which one is better. We can make it configurable when lowerings. 

# Spensor to Tensor

# Append Constant

`AppendConstant` is a helper class for generating constant index from `arith.constantOp` at once. Typically,
when generating `scf.parallel` and `affine.apply`, they use `arith.constant` as lower/upper bounds or offsets. 
A problem here is that they need to manually write construction of instances and a same constant could be repeated 
in different places. As a result, this pass manage the problem and remove the visual noise.

When some pass needs a index constant, it calls `getConstantOpByIndex` at `spensor_util.py` and get the constant operation.
The function `getConstantOpByIndex` receives the index, creates an operation if it is not in `index_to_constant_op` in 
`spensor_global.py`. By this way, it ensures usages of the same constant refer to the same operation.

By the end of lowering passes, this pass must be called to insert operations saved in `index_to_constant_op` to the 
beginning of a function. Otherwise, operations use constants on the fly.


# Spensor Dead Code Elimination

`SpensorDeadCodeElimination` is used for erasing some useless operations at the end of all lowering passes.
For example, `spensor.declareMemoryOp` is used only in `MemoryAnalysis` and they are not required by other 
operations. When erasing an operation, it ensures there is no dependency on the operation. 
