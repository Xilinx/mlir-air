# Spensor

This repository contains the implementation of Spensor dialect.

Here is the project structure:
```
+ spensor
  + spensor
    + cli (tools similar to mlir-opt to be used)
    + dialects (definition of Spensor dialect)
    + passes (include lowering from Spensor to different dialects)
    + utils (util functions used in lowerings)
```

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

## Installation
...

## Writing Spensor programs
...

## Extending Spensor Dialect
To add a new operation, here is the list we should do:
1. Add the operation under dialects 
2. Add support in lowering passes, typically, you want
  1. Add the operation pattern in parallel_analysis so it can compute the parallel upper bounds
  2. Add the operation pattern in expand_parallel so it generates the operation on Spensor type
  3. Add the operation pattern in spensor_to_tensor to generate the linalg op

You can also refer to docs/passes.md to see how each pass works

