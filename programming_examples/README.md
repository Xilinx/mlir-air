# MLIR-AIR Programming Examples

These programming examples are provided so that application programmers can learn how to leverage the AIR design flow with mlir-air python bindings, as well as the mlir-air intermediate representation (IR) directly, to build applications targeting AI Engines.

## [2-Dimensional Shim DMA Passthrough](shim_dma_2d)

This example demonstrates how data may be moved using shim DMA operations. It also includes extra infrastructure that illustrates different ways to compile, build, run, and test programs written using the mlir-air python bindings on an NPU.

## [Passthrough Examples](passthrough)

This directory contains three examples that each copy data from the input to the output (a data passthrough). The data movement is done through either DMA or Channels, and there is a simple example of calling a an external function which performs a vectorized memcopy.

## [Channel Examples](channel_examples)

This is a collection of simple examples that illustrate how to use *channels*. At a high level, channels are the abstraction for data movement in mlir-air. Some of the examples are experimental works-in-progress.

## [Matrix Scalar Addition](matrix_scalar_add)

This example provides logic to divide an input 2D matrix into *tiles* of data, and add a value to every element in every tile. It includes some description of the fundamental concepts of mlir-air, including *launches*, *herds*, and *channels*. There are five different implementations of this example, some of which are experimental (and are currently works-in-progress).

## [Data Transfer Transpose](data_transfer_transpose)

Transposes a matrix with using either air channels or `dma_memcpy_nd`.

## [Segment Alloc](segment_alloc)

While a *worker* (a compute unit managed as part of a *herd*) are able to allocate L1 memory, they are not able to allocate L2 memory. This must be done in the *segment*. This example shows how a segment can allocate L2 memory which is then accessed within the herd.

## [WIP: Multi-Segment Examples](multi_segment)

This is a collection of simple examples that illustrate how to use multiple segments.

Warning: This example is a work-in-progress.
