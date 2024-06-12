# MLIR-AIR Programming Examples

These programming examples are provided so that application programmers can learn how to leverage the AIR design flow with mlir-air python bindings, as well as the mlir-air intermediate representation (IR) directly, to build applications targeting AI Engines.

## [2-Dimensional Shim DMA Passthrough](shim_dma_2d)

This example demonstrates how data may be moved using shim DMA operations. It also includes extra infrastructure that illustrates different ways to compile, build, run, and test programs written using the mlir-air python bindings.

## [Matrix Scalar Addition](matrix_scalar_add)

This example provides logic to add one to every element of a matrix using tiling. It includes some description of the fundamental concepts of mlir-air, including *launches*, *segments*, *herds*, and *channels*.
