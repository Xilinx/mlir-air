# 2d_shim_dma

This example demonstrates how data may be moved using shim DMA operations. Specifically, in this example a 2-dimensional block of data (sometimes referred to in the code as an *image*) is set to have some specific values.

It was transferred to a compute core using DMA. The compute core then modifies an upper corner of the data (sometimes referred to in code as the *tile*).

The logic in this example is defined in [2d_shim_dma.py](2d_shim_dma.py) and is written using the MLIR-AIR python bindings. To compile and run the design for NPU:
```bash
make
make run
```