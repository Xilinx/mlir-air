#!/usr/bin/env python3
# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Test script that uses XRTRunner to compile, run, and validate the
multi-launch PDI reconfiguration example using ELF output format.
"""

import numpy as np
from air.backend.xrt_runner import XRTRunner
from air.ir import *

# Define the AIR module with two air.launch operations using iteration spaces
# - Launch 1 (add_two): iterates 8 times, processing tiles at offsets 0,16,32,...,112 (adds 2)
# - Launch 2 (add_three): iterates 8 times, processing tiles at offsets 128,144,160,...,240 (adds 3)
# This demonstrates PDI reconfiguration between launches while using iteration spaces
# to operate on various places in the host memory.

air_tiled_ir_string = """
module {
  func.func @reconfigure_example(%arg0: memref<512xi32>, %arg1: memref<512xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    // ========================================
    // Launch 1: @add_two design with 8 iterations
    // Each iteration processes a tile of 16 elements
    // Iteration x processes elements at offset x*16 (0, 16, 32, ..., 112)
    // Adds 2 to each element
    // ========================================
    air.launch (%x) in (%sz=%c8) args(%input=%arg0, %output=%arg1) : memref<512xi32>, memref<512xi32> attributes {id = 1 : i32} {
      %c1_0 = arith.constant 1 : index
      %c16_0 = arith.constant 16 : index
      %tile_offset = arith.muli %x, %c16_0 : index

      air.segment @add_two args(%seg_input=%input, %seg_output=%output, %offset=%tile_offset) : memref<512xi32>, memref<512xi32>, index attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index

        air.herd @herd_add_two tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%herd_input=%seg_input, %herd_output=%seg_output, %herd_offset=%offset) : memref<512xi32>, memref<512xi32>, index attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c16_h = arith.constant 16 : index
          %c2_i32 = arith.constant 2 : i32

          // L1 (tile memory) buffers - 16 elements per tile
          %l1_in = memref.alloc() : memref<16xi32, 2>
          %l1_out = memref.alloc() : memref<16xi32, 2>

          // DMA from input memory at dynamic offset into L1
          air.dma_memcpy_nd (%l1_in[] [] [], %herd_input[%herd_offset] [%c16_h] [%c1_h]) : (memref<16xi32, 2>, memref<512xi32>)

          // Add 2 to each element
          scf.for %i = %c0_h to %c16_h step %c1_h {
            %val = memref.load %l1_in[%i] : memref<16xi32, 2>
            %result = arith.addi %val, %c2_i32 : i32
            memref.store %result, %l1_out[%i] : memref<16xi32, 2>
          }

          // DMA from L1 to output memory at dynamic offset
          air.dma_memcpy_nd (%herd_output[%herd_offset] [%c16_h] [%c1_h], %l1_out[] [] []) : (memref<512xi32>, memref<16xi32, 2>)

          memref.dealloc %l1_in : memref<16xi32, 2>
          memref.dealloc %l1_out : memref<16xi32, 2>
        }
      }
    }

    // ========================================
    // Launch 2: @add_three design with 8 iterations (RECONFIGURATION)
    // Each iteration processes a tile of 16 elements
    // Iteration x processes elements at offset 128 + x*16 (128, 144, 160, ..., 240)
    // Adds 3 to each element (different computation than Launch 1)
    // ========================================
    air.launch (%x) in (%sz=%c8) args(%input=%arg0, %output=%arg1) : memref<512xi32>, memref<512xi32> attributes {id = 4 : i32} {
      %c1_0 = arith.constant 1 : index
      %c16_0 = arith.constant 16 : index
      %c128_0 = arith.constant 128 : index
      %iter_offset = arith.muli %x, %c16_0 : index
      %tile_offset = arith.addi %iter_offset, %c128_0 : index

      air.segment @add_three args(%seg_input=%input, %seg_output=%output, %offset=%tile_offset) : memref<512xi32>, memref<512xi32>, index attributes {id = 5 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index

        air.herd @herd_add_three tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%herd_input=%seg_input, %herd_output=%seg_output, %herd_offset=%offset) : memref<512xi32>, memref<512xi32>, index attributes {id = 6 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c16_h = arith.constant 16 : index
          %c3_i32 = arith.constant 3 : i32

          // L1 (tile memory) buffers - 16 elements per tile
          %l1_in = memref.alloc() : memref<16xi32, 2>
          %l1_out = memref.alloc() : memref<16xi32, 2>

          // DMA from input memory at dynamic offset into L1
          air.dma_memcpy_nd (%l1_in[] [] [], %herd_input[%herd_offset] [%c16_h] [%c1_h]) : (memref<16xi32, 2>, memref<512xi32>)

          // Add 3 to each element (different computation)
          scf.for %i = %c0_h to %c16_h step %c1_h {
            %val = memref.load %l1_in[%i] : memref<16xi32, 2>
            %result = arith.addi %val, %c3_i32 : i32
            memref.store %result, %l1_out[%i] : memref<16xi32, 2>
          }

          // DMA from L1 to output memory at dynamic offset
          air.dma_memcpy_nd (%herd_output[%herd_offset] [%c16_h] [%c1_h], %l1_out[] [] []) : (memref<512xi32>, memref<16xi32, 2>)

          memref.dealloc %l1_in : memref<16xi32, 2>
          memref.dealloc %l1_out : memref<16xi32, 2>
        }
      }
    }

    return
  }
}
"""


def main():
    DATA_COUNT = 512
    TILE_SIZE = 16
    NUM_ITERS = 8

    # Create input data: values 0-255 cycling
    input_data = np.array([i % 256 for i in range(DATA_COUNT)], dtype=np.int32)

    # Compute expected output:
    # Launch 1 (add_two): 8 iterations, each processing 16 elements at offsets 0,16,32,...,112
    # Launch 2 (add_three): 8 iterations, each processing 16 elements at offsets 128,144,160,...,240
    # Rest: zeros (not touched by the kernel)
    expected_output = np.zeros(DATA_COUNT, dtype=np.int32)

    # Launch 1: add 2 to elements 0-127 (8 tiles of 16 elements each)
    for i in range(NUM_ITERS):
        start = i * TILE_SIZE
        end = start + TILE_SIZE
        expected_output[start:end] = input_data[start:end] + 2

    # Launch 2: add 3 to elements 128-255 (8 tiles of 16 elements each)
    for i in range(NUM_ITERS):
        start = 128 + i * TILE_SIZE
        end = start + TILE_SIZE
        expected_output[start:end] = input_data[start:end] + 3

    print(f"Input (elements 0-15): {input_data[:16]}")
    print(f"Input (elements 128-143): {input_data[128:144]}")
    print(f"Expected output (elements 0-15, +2): {expected_output[:16]}")
    print(f"Expected output (elements 128-143, +3): {expected_output[128:144]}")
    print(f"Expected output (elements 256-271, unchanged): {expected_output[256:272]}")

    # Parse the AIR module
    with Context() as ctx, Location.unknown():
        air_module = Module.parse(air_tiled_ir_string)

        # Create XRTRunner with ELF output format
        # instance_name should match the func.func name (@reconfigure_example)
        runner = XRTRunner(
            output_format="elf",
            instance_name="reconfigure_example",  # matches func.func @reconfigure_example
            omit_while_true_loop=False,
        )

        # Run the test
        result = runner.run_test(
            mlir_module=air_module,
            inputs=[input_data],
            expected_outputs=[expected_output],
        )

        return result


if __name__ == "__main__":
    import sys

    sys.exit(main())
