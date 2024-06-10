# MLIR-AIR Programming Examples

These programming examples are provided so that application programmers can learn how to leverage the AIR design flow with mlir-air python bindings, and the mlir-air intermediate representation directly to build applications targeting AI Engines.

## [Matrix Scalar Addition](matrix_scalar_add)

This example provides logic to add one to every element of a matrix using tiling. It includes some description of the fundamental concepts of AIR, including launches, segments, herds, and channels.

## [2-Dimensional Shim DMA Passthrough](shim_dma_2d)

This example demonstrates how data may be moved using shim DMA operations. It also includes extra tooling that illustrates different ways to compile, build, run, and test code defined as MLIR-AIR Python bindings on an NPU.
