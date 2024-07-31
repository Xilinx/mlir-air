<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// 
//===----------------------------------------------------------------------===//-->

# WIP: Multi-Segment Examples

These two examples are attempts to create AIR programs using multiple segments. For simplicity, the segments are logically independent (they do not communicate with each other).

Warning! Neither of the examples are functional. The design has been checked in each case by running it with one segment or the other, but in it's entirety, multiple segments are not yet supported.

## Usage

### Generate AIR MLIR from Python

Run:
```bash
make print
```
OR 

```bash
python multi_segment.py -p
```

### Running

To compile and run the design:

```bash
make
```

To run with verbose output, either modify the makefile or specify verbose directly with:
```bash
python multi_segment.py -v
```
