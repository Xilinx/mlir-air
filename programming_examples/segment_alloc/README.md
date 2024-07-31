<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// 
//===----------------------------------------------------------------------===//-->

# Segment L2 Allocation Example

This example is functionally a data passthrough, but what makes it interesting is that some of the data accessed in a herd is L2 data allocated by a segment. This L2 memory is passed as an argument into the herd.

## Source Files Overview

1. [`segment_alloc.py`](./segment_alloc.py): Python scripts that defines the module design for the example using MLIR-AIR Python bindings. The file generates MLIR that is then compiled using `aircc.py` to produce design binaries (i.e. `XCLBIN` and `inst.txt` for the NPU in Ryzen™ AI). This file also contains the code needed to run the program on the NPU and test the output.

## Usage

### Generate AIR MLIR from Python

Run:
```bash
make print
```
OR 

```bash
python segment_alloc.py -p
```

### Running

To compile and run the design:

```bash
make
```

To run with verbose output, either modify the makefile or specify verbose directly with:
```bash
python segment_alloc.py -v
```
