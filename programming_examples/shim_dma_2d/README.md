# shim_dma_2d

This example demonstrates how data may be moved using shim DMA operations. Specifically, in this example a 2-dimensional block of data (sometimes referred to in the code as an *image*) is set to have some specific values.

It was transferred to a compute core using DMA. The compute core then modifies an upper corner of the data (sometimes referred to in code as the *tile*).

The logic in this example is defined in [shim_dma_2d.py](shim_dma_2d.py).

For illustrative purposes, there are 3 ways to run and test this example.

## Run and test with AIR utility functions

This is the cleanest and simplest method of running MLIR-AIR code on NPUs, and uses code in the [run.py](run.py) file.

```bash
make pyworkflow
```

## Generate MLIR-AIR with python, compile on the command line, and run with python

This method uses the [test.py](test.py) file. This file is included for better understanding of what the utility functions avialable by the XRT air backend do behind the scenes.

```bash
make
make run_py
```

## Generate MLIR with python, compile on the command line, and run with C++

This method uses the [test.cpp](test.cpp) file. While python is very clean, this example is used to show how C++ may be used.

```bash
make
make run
```