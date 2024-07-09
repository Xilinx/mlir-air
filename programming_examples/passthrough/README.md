<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// 
//===----------------------------------------------------------------------===//-->

# Passthrough Kernel:

This example, called "Passthrough", demonstrates a simple MLIR-AIR implementation for copying a vector of bytes. In this design, a single AIR worker core performs the memcpy operation on a vector with a default length `4096`. The copy operation is a `1024` element-sized subvector and is invoked multiple times to complete the full copy. The example consists of three methods of doing this: [`passthrough_dma`](./passthrough_dma) which uses DMA for data movement and a simple per-element copy operation, [`passthrough_channel`](./passthrough_channel) which uses channels for data movement and the same simple per-element copy operations, and [`passthrough_kernel`](./passthrough_kernel) which calls an external function that performs a vectorized copy operation on the subvector.

## Source Files Overview

1. [`passthrough_dma/passthrough_dma.py`](./passthrough_dma/passthrough_dma.py), [`passthrough_channel/passthrough_channel.py`](passthrough_channel/passthrough_channel.py), [`passthrough_kernel/passthrough_kernel.py`](passthrough_kernel/passthrough_kernel.py): Python scripts that defines the module design for each example using MLIR-AIR Python bindings. The file generates MLIR that is then compiled using `aircc.py` to produce design binaries (i.e. `XCLBIN` and `inst.txt` for the NPU in Ryzen™ AI). You can run `python passthrough_(dma|channel|kernel).py` to generate the MLIR.

1. `passThrough.cc`: A C++ implementation of vectorized memcpy operations for AIE cores. It is found in the [mlir-aie repo](https://github.com/Xilinx/mlir-aie) under [`mlir-aie/aie_kernels/generic/passThrough.cc`](https://github.com/Xilinx/mlir-aie/blob/main/aie_kernels/generic/passThrough.cc)

1. `run.py` files and `common.py`: This Python code is a testbench for the Passthrough design examples. The code is responsible for compiling and loading the compiled XCLBIN file, configuring the AIR module, providing input and output data, and executing the AIR design on the NPU. After executing, the script verifies the memcpy results.

## Design Overview

See the [design overview](https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/passthrough_kernel) in the [mlir-aie repo](https://github.com/Xilinx/mlir-aie) for more information!

## Usage

### Running

To compile and run the design:

```
make
```