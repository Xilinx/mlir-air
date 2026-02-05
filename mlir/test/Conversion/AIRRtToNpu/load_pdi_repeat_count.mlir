//===- load_pdi_repeat_count.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test the generation of aiex.npu.load_pdi at air.launch_end locations.
// load_pdi is generated when BOTH conditions are true:
// 1. emit-main-device=true is set
// 2. The device has core/memtile DMAs with repeat_count > 0

// RUN: air-opt -airrt-to-npu="emit-main-device=true" --split-input-file %s | FileCheck %s --check-prefix=EMIT-TRUE
// RUN: air-opt -airrt-to-npu="emit-main-device=false" --split-input-file %s | FileCheck %s --check-prefix=EMIT-FALSE

// Test 1: npu2 device with core DMA repeat_count > 0
// With emit-main-device=true: load_pdi SHOULD be generated inside <device>_sequence
// With emit-main-device=false: load_pdi should NOT be generated

// EMIT-TRUE-LABEL: aie.device(npu2) @segment0 {
// EMIT-TRUE: aie.runtime_sequence @segment0_sequence
// EMIT-TRUE:   aiex.dma_configure_task_for @airMemcpyId7 {
// EMIT-TRUE:   aiex.dma_start_task
// EMIT-TRUE:   aiex.dma_await_task
// EMIT-TRUE:   aiex.npu.load_pdi {device_ref = @segment0}
// EMIT-TRUE: }

// EMIT-FALSE-LABEL: aie.device(npu2) @segment0 {
// EMIT-FALSE: aie.runtime_sequence @func_with_repeat_count
// EMIT-FALSE:   aiex.dma_configure_task_for @airMemcpyId7 {
// EMIT-FALSE:   aiex.dma_start_task
// EMIT-FALSE:   aiex.dma_await_task
// EMIT-FALSE-NOT:   aiex.npu.load_pdi
// EMIT-FALSE: }

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId7(%shim_noc_tile_0_0, S2MM, 0)
    
    // Core DMA with repeat_count > 0
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 3)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "segment0"}
  
  airrt.module_metadata{}
  
  func.func @func_with_repeat_count(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c7_i32 = arith.constant 7 : i32
    %0 = airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment0" : i64
    return
  }
}

// -----

// Test 2: npu2 device with NO DMAs with repeat_count > 0
// load_pdi should NOT be generated regardless of emit-main-device setting

// EMIT-TRUE-LABEL: aie.device(npu2) @segment_no_repeat {
// EMIT-TRUE: aie.runtime_sequence @segment_no_repeat_sequence
// EMIT-TRUE:   aiex.dma_configure_task_for @airMemcpyId8 {
// EMIT-TRUE:   aiex.dma_start_task
// EMIT-TRUE:   aiex.dma_await_task
// EMIT-TRUE-NOT:   aiex.npu.load_pdi
// EMIT-TRUE: }

// EMIT-FALSE-LABEL: aie.device(npu2) @segment_no_repeat {
// EMIT-FALSE: aie.runtime_sequence @func_no_repeat
// EMIT-FALSE:   aiex.dma_configure_task_for @airMemcpyId8 {
// EMIT-FALSE:   aiex.dma_start_task
// EMIT-FALSE:   aiex.dma_await_task
// EMIT-FALSE-NOT:   aiex.npu.load_pdi
// EMIT-FALSE: }

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId8(%shim_noc_tile_0_0, S2MM, 0)
    
    // Core DMA with repeat_count = 0 (no repeat)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "segment_no_repeat"}
  
  airrt.module_metadata{}
  
  func.func @func_no_repeat(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c8_i32 = arith.constant 8 : i32
    %0 = airrt.dma_memcpy_nd(%c8_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId8} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment_no_repeat" : i64
    return
  }
}

// -----

// Test 3: npu1 device should NOT get load_pdi even with repeat_count > 0
// and emit-main-device=true (only NPU2 family devices get load_pdi)

// EMIT-TRUE-LABEL: aie.device(npu1_1col) @segment_npu1 {
// EMIT-TRUE: aie.runtime_sequence @segment_npu1_sequence
// EMIT-TRUE:   aiex.dma_configure_task_for @airMemcpyId9 {
// EMIT-TRUE:   aiex.dma_start_task
// EMIT-TRUE:   aiex.dma_await_task
// EMIT-TRUE-NOT:   aiex.npu.load_pdi
// EMIT-TRUE: }

// EMIT-FALSE-LABEL: aie.device(npu1_1col) @segment_npu1 {
// EMIT-FALSE: aie.runtime_sequence @func_npu1
// EMIT-FALSE:   aiex.dma_configure_task_for @airMemcpyId9 {
// EMIT-FALSE:   aiex.dma_start_task
// EMIT-FALSE:   aiex.dma_await_task
// EMIT-FALSE-NOT:   aiex.npu.load_pdi
// EMIT-FALSE: }

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId9(%shim_noc_tile_0_0, S2MM, 0)
    
    // Core DMA with repeat_count > 0
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 3)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "segment_npu1"}
  
  airrt.module_metadata{}
  
  func.func @func_npu1(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c9_i32 = arith.constant 9 : i32
    %0 = airrt.dma_memcpy_nd(%c9_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId9} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment_npu1" : i64
    return
  }
}

// -----

// Test 4: wait_all WITHOUT air.launch_end attribute should NOT get load_pdi
// regardless of emit-main-device setting

// EMIT-TRUE-LABEL: aie.device(npu2) @segment_no_launch_end {
// EMIT-TRUE: aie.runtime_sequence @segment_no_launch_end_sequence
// EMIT-TRUE:   aiex.dma_configure_task_for @airMemcpyId10 {
// EMIT-TRUE:   aiex.dma_start_task
// EMIT-TRUE:   aiex.dma_await_task
// EMIT-TRUE-NOT:   aiex.npu.load_pdi
// EMIT-TRUE: }

// EMIT-FALSE-LABEL: aie.device(npu2) @segment_no_launch_end {
// EMIT-FALSE: aie.runtime_sequence @func_no_launch_end
// EMIT-FALSE:   aiex.dma_configure_task_for @airMemcpyId10 {
// EMIT-FALSE:   aiex.dma_start_task
// EMIT-FALSE:   aiex.dma_await_task
// EMIT-FALSE-NOT:   aiex.npu.load_pdi
// EMIT-FALSE: }

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId10(%shim_noc_tile_0_0, S2MM, 0)
    
    // Core DMA with repeat_count > 0
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 3)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "segment_no_launch_end"}
  
  airrt.module_metadata{}
  
  func.func @func_no_launch_end(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c10_i32 = arith.constant 10 : i32
    %0 = airrt.dma_memcpy_nd(%c10_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    // Regular wait_all without air.launch_end
    airrt.wait_all %0
    %p = airrt.segment_load "segment_no_launch_end" : i64
    return
  }
}

// -----

// Test 5: memtile DMA with repeat_count > 0 should also trigger load_pdi
// when emit-main-device=true

// EMIT-TRUE-LABEL: aie.device(npu2) @segment_memtile {
// EMIT-TRUE: aie.runtime_sequence @segment_memtile_sequence
// EMIT-TRUE:   aiex.dma_configure_task_for @airMemcpyId11 {
// EMIT-TRUE:   aiex.dma_start_task
// EMIT-TRUE:   aiex.dma_await_task
// EMIT-TRUE:   aiex.npu.load_pdi {device_ref = @segment_memtile}
// EMIT-TRUE: }

// EMIT-FALSE-LABEL: aie.device(npu2) @segment_memtile {
// EMIT-FALSE: aie.runtime_sequence @func_memtile_repeat
// EMIT-FALSE:   aiex.dma_configure_task_for @airMemcpyId11 {
// EMIT-FALSE:   aiex.dma_start_task
// EMIT-FALSE:   aiex.dma_await_task
// EMIT-FALSE-NOT:   aiex.npu.load_pdi
// EMIT-FALSE: }

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId11(%shim_noc_tile_0_0, S2MM, 0)
    
    // Memtile DMA with repeat_count > 0
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 5)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "segment_memtile"}
  
  airrt.module_metadata{}
  
  func.func @func_memtile_repeat(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c11_i32 = arith.constant 11 : i32
    %0 = airrt.dma_memcpy_nd(%c11_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId11} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment_memtile" : i64
    return
  }
}
