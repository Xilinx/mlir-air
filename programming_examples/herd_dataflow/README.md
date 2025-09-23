# AIR Herd Dataflow Example

This tutorial demonstrates how to design and implement a dataflow design using [mlir-air](https://github.com/Xilinx/mlir-air). The example is intended for users who are familiar with MLIR but new to mlir-air, and provides both an MLIR-based and a Python-based implementation of a tiled, multi-stage dataflow design.

## Overview

The example models a tiled dataflow computation mapped to a 4-column AIE array.

- **What you'll build**: a three‑herd pipeline that (1) brings tiles from DDR via L2 memory to L1, (2) performs vector add with the top row of 4 compute cores (i.e. the first air.herd), (3) copies data via the second row of 4 compute cores (i.e. the second air.herd), (4) calls an external kernel which performs "add 3" with the third row of 4 compute cores (i.e. the third air.herd), and (5) returns results to DDR via L2. The Python script constructs the AIR IR, runs it, and checks `C := A + B + 3`.

The design is provided in two forms:
- `air.mlir`: The mlir-air dialect implementation
- `run.py`: A Python implementation using mlir-air Python bindings

The build and execution flow is managed by the provided `Makefile`.

## Directory Structure

```
herd_dataflow/
├── air.mlir         # mlir-air implementation of the design
├── run.py           # Python bindings implementation (mlir-air Python API)
├── Makefile         # Build and run automation, parameterized for problem size
├── README.md        # This file
├── extern_func.cc   # External C++ function for the third air.herd
```

## Key AIR Constructs in This Design

This example demonstrates several key abstractions in mlir-air for expressing dataflow and hardware mapping:

### 1. Data Movement: `air.dma_memcpy_nd` vs. `air.channel`

Two forms of data movement are used:
- **`air.dma_memcpy_nd`**: Directly describes multi-dimensional DMA transfers between memory spaces (e.g., L2 to L1, L1 to L2). This is a synchronous, explicit data movement operation, with source and destination coupled together.
- **`air.channel`**: Declares communication channels between regions of the design. Channels decouple data movement from computation, allowing the compiler to schedule data movement and compute independently. This decoupling exposes locality and enables back-pressure, guaranteeing correctness even when data movement and compute are overlapped.

Channels can be configured with different `channel_type` attributes, to represent (1) circuit-switched, (2) packet-switched or (3) cascase (peer-to-peer) data movement mechanisms supported by the target device.

By placing `put`/`get` operations at the correct hierarchy boundaries, the design exposes opportunities for the compiler to optimize and overlap communication and computation.

### 2. Resource Scoping & Dispatch: `air.launch`, `air.segment`, and `air.herd`

- **`air.launch`**: Offloads a region of code to the accelerator. Its iteration space is managed by the runtime and can be scheduled in parallel or in sequence.
- **`air.segment`**: Reserves a resource pool (such as a region of tiles or memory) for the nested work, providing explicit scoping of resources.
- **`air.herd`**: Defines a group of compute tiles working in parallel to execute the same kernel. In this design, each `herd` represents a stage in the pipeline, mapped to a row of compute tiles in the hardware array.
- **`scf.parallel` / `scf.forall`**: Represents parallelism in code regions, such as parallel data movement. In this example, the use of `scf.parallel` (or `scf.forall`) provides the compiler with more information about how data movement can be partitioned and executed concurrently, enabling better optimization and mapping to hardware.

At the top level, the example creates a tiled launch space and a segment that owns resources for the pipeline, then instantiates three herds for the three pipeline stages.

### 3. Kernel Formats

- **First herd**: Demonstrates vectorized computation using MLIR's `vector` and `memref` dialect operations for efficient parallel computation.
- **Second herd**: Demonstrates a scalar code format, showing how to write simple, non-vectorized kernels.
- **Third herd**: Calls an external function, compiled from an external C++ project, demonstrating how to integrate custom compute kernels into the AIR pipeline.

These constructs together enable the design to express complex, tiled, and pipelined dataflow computations that map efficiently to AIE hardware.

## Key Files

- **air.mlir**: The main MLIR file, written in the mlir-air dialect, expressing the full dataflow and hardware mapping.
- **run.py**: Python script using mlir-air Python bindings to build and run the same design. Useful for rapid prototyping and experimentation.
- **Makefile**: Automates building and running the example, with parameterizable problem sizes.

## How to Use

### Prerequisites

- Mlir-air installed and in your environment
- Python with required packages (see mlir-air documentation)

### Building and Running

You can build and run the example using the Makefile. The main targets are:

- `make run`: Build and run the example with default parameters.
- `make print`: Print the generated MLIR IR for inspection.

#### Parameterizing Problem Size

You can override the default problem size by specifying `M_SIZE` and `N_SIZE`:

```sh
make run M_SIZE=1024 N_SIZE=1024
```

**Note:**  
- `M_SIZE` must be a multiple of 64  
- `N_SIZE` must be a multiple of 256 (assuming 4 columns of cores)  
Partial tiles are not supported.

#### Inspecting the IR

To print the generated MLIR IR (without running the computation):

```sh
make print
```

#### Cleaning Up

To remove build artifacts:

```sh
make clean
```

## Explanation of Key Parameters

- **M_SIZE**: Number of rows in the input matrix
- **N_SIZE**: Number of columns in the input matrix
- **NUM_COLUMNS**: Number of columns in the hardware array
- **L1_BUFFER_SIZE_M/N**: Tile-local buffer sizes in number of elements
- **L2_BUFFER_SIZE_M/N**: Shared buffer sizes in number of elements
- **VECTOR_SIZE**: Number of elements processed in a single vector operation (16, matches hardware vector width)
- **CHANNEL_STRIDE**: Stride in the column dimension for channel and DMA operations (1 = contiguous)

## Customizing the Example

- You can modify `air.mlir` to experiment with new designs in *mlir-air* dialect.
- You can modify `run.py` to experiment with new designs using the *Python API*.

## Further Reading

- [mlir-air GitHub](https://github.com/Xilinx/mlir-air)
- [mlir-aie GitHub](https://github.com/Xilinx/mlir-aie)
- [MLIR Documentation](https://mlir.llvm.org/)
- [AI Engine Documentation](https://docs.xilinx.com/r/en-US/ug1076-ai-engine-environment)

## Support

For questions or issues, please open an issue on the [mlir-air GitHub](https://github.com/Xilinx/mlir-air) repository.
