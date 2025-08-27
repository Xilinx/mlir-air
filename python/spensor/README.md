# Spensor

Spatial Tensor (Spensor) is designed to support explicit data movements across memory hierarchies.
This repository contains the implementation of Spensor dialect, and compilation passes from Spensor
to memrefs.

Here is the project structure:
```
+ spensor
  + spensor
    + cli (tools similar to mlir-opt to be used)
    + dialects (definition of Spensor dialect)
    + passes (include lowering from Spensor to different dialects)
    + utils (util functions used in lowerings)
```

## What can you do with Spensor?

A Spensor is a regular tensor attached with memory location information such as `<<2x2xf32>, L1>` 
or `<<4xi8>, L3>`. A NDSpensor orgranizes a group of Spensor such as `<4x<2x2xf32, L1>>` meaning
that there are four `<2x2xf32>` located at `L1` memory. Because Spensor type carries memory location, 
Spensor dialect supports movement operations like `move(<<2x2xf32>, L1>) to L2`. In addition, Spensor 
dialect offers operations manipulating `NDSpensor` such as combining or reduce a `NDSpensor` to a
smaller one. 

## Installation

Suppose you are at `mlir-air`, all you need is
```
pip install python/spensor
```

Or provide the path to `spensor` if you are in a different folder.

## Lowerings

Here is the expected lowering passes from Spensor to MLIR core dialects:
```
Operation on NDSpensors
        | (Memory and loop analysis, expand into parallel loops and tiling)
        V
Operation on Spensors wrapped by parallel loops
        | (Spensor to tensor)
        V
Operation on tensor types, a mix of tensor, scf, bufferization, and linalg dialects
        | (mlir-opt -one-shot-bufferize)
        V
Operation on memref types, ready to be sent to MLIR-AIR
```

## Writing Spensor programs

First, the user should specify memories supported the architecture.
```
L1_memory = spensor.MemoryType("L1", [6])
L2_memory = spensor.MemoryType("L2", [])
l3_memory = spensor.MemoryType("L3", [])
declare_l1 = spensor.DeclareMemoryOp(
    "L1", L1_memory.memory_shape, ["L2", "L3"], ["L2", "L3"], 2
)
declare_l2 = spensor.DeclareMemoryOp(
    "L2", L2_memory.memory_shape, ["L3"], ["L3"], 1
)
declare_l3 = spensor.DeclareMemoryOp("L3", l3_memory.memory_shape, [], [])
```

At the example above, we declared there are `L1, L2, and L3` memories.

Next, we declare a function for the Spensor computation:
```
spensor_input_type = spensor.SpensorType(TensorType(f32, [16, 32]), l3_memory)
spensor_output_type = spensor.SpensorType(TensorType(f32, [16, 1]), l3_memory)
function_type = FunctionType.from_lists(
  [
      spensor_input_type,
  ],
  [spensor_output_type],
)
# function(arg0: <<16x32xf32>, L3>) -> <<16x1xf32>, L1>
func_op = func.FuncOp("reduce", function_type)
```

and, we can do more operations based on function arguments and returns the value:


```
split_arg0 = spensor.SplitOp(arg0, num_partitions=const_4, dim=const_1)
arg0_l2 = spensor.MoveOp(split_arg0, L2_memory)

arg0_l1 = spensor.MoveOp(arg0_l2, L1_memory)
reduce_res = spensor.ReduceSumOp(arg0_l1)

combine_res = spensor.NDCombineOp(reduce_res, nd_dim=const_0, dim=const_1, memory=L2_memory)
combine_res_l1 = spensor.MoveOp(combine_res, L1_memory)
combine_reduce_res = spensor.ReduceSumOp(combine_res_l1)

combine_reduce_res_l2 = spensor.MoveOp(combine_reduce_res.result, L2_memory)

reduce_res_l3 = spensor.MoveOp(combine_reduce_res_l2.result, l3_memory)
result = spensor.CombineToSpensorOp(reduce_res_l3.result, (0,))
return_op = func.ReturnOp(result.result)
```

In `python/test/spensor/` you can find more examples.

## Extending Spensor operations
To add a new operation, here is the list we should do:
1. Add the operation into `spensor_dialect.py` 
2. Add support in lowering passes, typically, you want
    1. Add the operation pattern in `parallel_analysis.py` so it can compute the parallel upper bounds
    2. Add the operation pattern in `expand_parallel.py` so it generates the operation on Spensor type
    3. Add the operation pattern in `spensor_to_tensor.py` to generate the linalg op

You can also refer to docs/passes.md to see how each pass works

