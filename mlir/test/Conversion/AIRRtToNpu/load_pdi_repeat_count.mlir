//===- load_pdi_repeat_count.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test the generation of aiex.npu.load_pdi at air.launch_end locations.
// load_pdi is generated when BOTH conditions are true:
// 1. output-elf=true is set (ELF output mode)
// 2. The device has core/memtile DMAs with repeat_count > 0

// RUN: air-opt -airrt-to-npu="output-elf=true" --split-input-file %s | FileCheck %s --check-prefix=EMIT-TRUE
// RUN: air-opt -airrt-to-npu="output-elf=false" --split-input-file %s | FileCheck %s --check-prefix=EMIT-FALSE

// Test 1: npu2 device with core DMA repeat_count > 0
// With output-elf=true: load_pdi SHOULD be generated inside <device>_sequence
// With output-elf=false: load_pdi should NOT be generated

// The reset device should exist with preserved DMA BDs and no
// runtime_sequence. It is a lightweight clone for between-iteration
// load_pdi that resets DMA/lock state without reloading ELFs.
// EMIT-TRUE-LABEL: aie.device(npu2) @segment0_reset {
// EMIT-TRUE:   aie.mem
// EMIT-TRUE:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 3)
// EMIT-TRUE-NOT: runtime_sequence
// EMIT-TRUE: }
// EMIT-TRUE-LABEL: aie.device(npu2) @segment0 {
// EMIT-TRUE: aie.runtime_sequence @func_with_repeat_count
// EMIT-TRUE:   aiex.dma_configure_task_for @airMemcpyId7 {
// EMIT-TRUE:   aiex.dma_start_task
// EMIT-TRUE:   aiex.dma_await_task
// EMIT-TRUE:   aiex.npu.load_pdi {device_ref = @segment0_reset}
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
    %0 = airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment0" : i64
    return
  }
}

// -----

// Test 2: npu2 device with NO DMAs with repeat_count > 0
// load_pdi should NOT be generated regardless of output-elf setting

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
    %0 = airrt.dma_memcpy_nd(%c8_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId8} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment_no_repeat" : i64
    return
  }
}

// -----

// Test 3: npu1 device should NOT get load_pdi even with repeat_count > 0
// and output-elf=true (only NPU2 family devices get load_pdi)

// EMIT-TRUE-LABEL: aie.device(npu1_1col) @segment_npu1 {
// EMIT-TRUE: aie.runtime_sequence @func_npu1
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
    %0 = airrt.dma_memcpy_nd(%c9_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId9} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment_npu1" : i64
    return
  }
}

// -----

// Test 4: wait_all WITHOUT air.launch_end attribute should NOT get load_pdi
// regardless of output-elf setting

// EMIT-TRUE-LABEL: aie.device(npu2) @segment_no_launch_end {
// EMIT-TRUE: aie.runtime_sequence @func_no_launch_end
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
    %0 = airrt.dma_memcpy_nd(%c10_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // Regular wait_all without air.launch_end
    airrt.wait_all %0
    %p = airrt.segment_load "segment_no_launch_end" : i64
    return
  }
}

// -----

// Test 5: memtile DMA with repeat_count > 0 should also trigger load_pdi
// when output-elf=true

// EMIT-TRUE-LABEL: aie.device(npu2) @segment_memtile {
// EMIT-TRUE: aie.runtime_sequence @func_memtile_repeat
// EMIT-TRUE:   aiex.dma_configure_task_for @airMemcpyId11 {
// EMIT-TRUE:   aiex.dma_start_task
// EMIT-TRUE:   aiex.dma_await_task
// EMIT-TRUE:   aiex.npu.load_pdi {device_ref = @segment_memtile_reset}
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
    %0 = airrt.dma_memcpy_nd(%c11_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId11} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment_memtile" : i64
    return
  }
}

// -----

// Test 6: Device with CoreOp — the reset device should have an empty
// CoreOp (no elf_file, no link_with, empty body) so that initLocks
// and addCoreEnable still fire during PDI expansion, but no ELF
// is compiled or loaded.

// The reset device has DMA BDs preserved, no runtime_sequence,
// and CoreOps replaced with empty shells (no elf_file/link_with).
// EMIT-TRUE-LABEL: aie.device(npu2) @segment_with_core_reset {
// EMIT-TRUE:   aie.mem
// EMIT-TRUE:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 3)
// EMIT-TRUE-NOT: link_with
// EMIT-TRUE-NOT: elf_file
// EMIT-TRUE-NOT: runtime_sequence
// EMIT-TRUE: }
// EMIT-TRUE-LABEL: aie.device(npu2) @segment_with_core {
// EMIT-TRUE:   aie.core
// EMIT-TRUE:   aie.runtime_sequence @func_with_core
// EMIT-TRUE:     aiex.npu.load_pdi {device_ref = @segment_with_core_reset}

// EMIT-FALSE-LABEL: aie.device(npu2) @segment_with_core {
// EMIT-FALSE-NOT: segment_with_core_reset
// EMIT-FALSE-NOT: aiex.npu.load_pdi

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId12(%shim_noc_tile_0_0, S2MM, 0)

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 3)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }

    // Core with link_with — should be replaced with empty core in reset device
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "segment_with_core"}

  airrt.module_metadata{}

  func.func @func_with_core(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c12_i32 = arith.constant 12 : i32
    %0 = airrt.dma_memcpy_nd(%c12_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId12} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment_with_core" : i64
    return
  }
}

// -----

// Test 7: No repeat_count, air.launch_end WaitAllOp without DMA operands
// When output-elf=true, NpuDmaWaitOp should be emitted for all shim channels
// to provide between-iteration synchronization (issue #1373).
// When output-elf=false, no sync is needed.

// EMIT-TRUE-LABEL: aie.device(npu2) @segment_no_repeat_no_dma_opers {
// EMIT-TRUE: aie.runtime_sequence @segment_no_repeat_no_dma_opers_sequence
// EMIT-TRUE:   aiex.dma_configure_task_for @airMemcpyId13 {
// EMIT-TRUE:   aiex.dma_start_task
// EMIT-TRUE:   aiex.dma_await_task
// EMIT-TRUE-NOT:   aiex.npu.load_pdi

// EMIT-FALSE-LABEL: aie.device(npu2) @segment_no_repeat_no_dma_opers {
// EMIT-FALSE: aie.runtime_sequence
// EMIT-FALSE:   aiex.dma_configure_task_for @airMemcpyId13 {
// EMIT-FALSE:   aiex.dma_start_task
// EMIT-FALSE-NOT:   aiex.dma_await_task
// EMIT-FALSE-NOT:   aiex.npu.load_pdi

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.shim_dma_allocation @airMemcpyId13(%tile_0_0, S2MM, 0)

    // Core DMA without repeat_count (infinite cycling)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "segment_no_repeat_no_dma_opers"}

  airrt.module_metadata{}

  func.func @func_no_repeat_no_dma_opers(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c13_i32 = arith.constant 13 : i32
    %0 = airrt.dma_memcpy_nd(%c13_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId13} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // WaitAllOp with air.launch_end but NO DMA operands
    // Pattern 1 won't match (no DMA operands), Pattern 2 handles it.
    // With output-elf=true, should emit NpuDmaWaitOp for @airMemcpyId13.
    airrt.wait_all {"air.launch_end"}
    %p = airrt.segment_load "segment_no_repeat_no_dma_opers" : i64
    return
  }
}

// -----

// Test 8: device with a cascade flow but NO repeat_count DMA (single-trip cascade).
// Cascade core-locks need the same per-launch reset as repeat_count DMAs -- they do not
// re-arm across host re-dispatch on their own -- so with output-elf=true load_pdi SHOULD
// be generated even though repeat_count == 0.

// EMIT-TRUE-LABEL: aie.device(npu2) @segment_cascade_reset {
// EMIT-TRUE-NOT: runtime_sequence
// EMIT-TRUE: }
// EMIT-TRUE-LABEL: aie.device(npu2) @segment_cascade {
// EMIT-TRUE: aie.runtime_sequence @func_cascade
// EMIT-TRUE:   aiex.npu.load_pdi {device_ref = @segment_cascade_reset}
// EMIT-TRUE: }

// EMIT-FALSE-LABEL: aie.device(npu2) @segment_cascade {
// EMIT-FALSE: aie.runtime_sequence @func_cascade
// EMIT-FALSE-NOT:   aiex.npu.load_pdi
// EMIT-FALSE: }

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.shim_dma_allocation @airMemcpyId14(%tile_0_0, S2MM, 0)
    aie.cascade_flow(%tile_0_2, %tile_0_3)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "segment_cascade"}
  airrt.module_metadata{}
  func.func @func_cascade(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c14_i32 = arith.constant 14 : i32
    %0 = airrt.dma_memcpy_nd(%c14_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId14} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0 {"air.launch_end"}
    %p = airrt.segment_load "segment_cascade" : i64
    return
  }
}

// -----

// Test 9: cascade flow whose air.launch_end sits inside a MULTI-ITERATION launch
// loop (repeat_count == 0). Unlike the single-trip cascade in Test 8, the cascade
// locks re-arm every iteration on their own, so the between-iteration load_pdi
// reset is unnecessary and costly (one PDI reload per boundary). load_pdi should
// NOT be generated, even with output-elf=true.

// EMIT-TRUE-LABEL: aie.device(npu2) @segment_cascade_loop {
// EMIT-TRUE: aie.runtime_sequence @func_cascade_loop
// EMIT-TRUE-NOT:   aiex.npu.load_pdi
// EMIT-TRUE: }

// EMIT-FALSE-LABEL: aie.device(npu2) @segment_cascade_loop {
// EMIT-FALSE: aie.runtime_sequence @func_cascade_loop
// EMIT-FALSE-NOT:   aiex.npu.load_pdi
// EMIT-FALSE: }

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.shim_dma_allocation @airMemcpyId15(%tile_0_0, S2MM, 0)
    aie.cascade_flow(%tile_0_2, %tile_0_3)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "segment_cascade_loop"}
  airrt.module_metadata{}
  func.func @func_cascade_loop(%arg0: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c15_i32 = arith.constant 15 : i32
    // Multi-iteration launch boundary: air.launch_end fires once per iteration.
    affine.for %arg1 = 0 to 2 {
      %0 = airrt.dma_memcpy_nd(%c15_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId15} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      airrt.wait_all %0 {"air.launch_end"}
    }
    %p = airrt.segment_load "segment_cascade_loop" : i64
    return
  }
}

// -----

// Test 10: FLASH-ATTENTION PREFILL REGRESSION GUARD.
//
// This mirrors the shape of flash-attention prefill's lowered control code at
// airrt-to-npu time: a cascade device (aie.cascade_flow) whose multi-iteration
// launch (the lq_iters loop) has already been UNROLLED, so it appears as several
// airrt.wait_all {air.launch_end} markers -- here two -- sitting inside a
// degenerate `affine.for 0 to 1` wrapper, each preceded by its own DMA loops.
// There is NO enclosing multi-trip loop, so a per-launch_end loop-trip check
// alone would misread this as single-trip and insert the load_pdi reset between
// the iterations (a silent ~42% latency regression: 708 -> 1004 us on NPU2 at
// LQ=LK=512, 2 heads). Because the cascade locks re-arm every iteration on their
// own, load_pdi must NOT be generated here even with output-elf=true. The guard:
// more than one air.launch_end on the device => multi-iteration => no reset.

// The costly reset-device clone (emitted before the segment device) must NOT be
// created, and no load_pdi must appear in the control sequence.
// EMIT-TRUE-NOT: @flash_attn_seg_reset
// EMIT-TRUE-LABEL: aie.device(npu2) @flash_attn_seg {
// EMIT-TRUE: aie.runtime_sequence @fa_ctrl
// EMIT-TRUE-NOT: aiex.npu.load_pdi
// EMIT-TRUE: }

// EMIT-FALSE-LABEL: aie.device(npu2) @flash_attn_seg {
// EMIT-FALSE: aie.runtime_sequence @fa_ctrl
// EMIT-FALSE-NOT: aiex.npu.load_pdi
// EMIT-FALSE: }

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.shim_dma_allocation @flashQKIn(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @flashOut(%tile_0_0, S2MM, 0)
    // Cascade bus between the two compute tiles (as in flash-attention).
    aie.cascade_flow(%tile_0_2, %tile_0_3)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      aie.end
    ^bb2:
      aie.end
    }
  } {sym_name = "flash_attn_seg"}
  airrt.module_metadata{}
  func.func @fa_ctrl(%q: memref<512x64xbf16>, %o: memref<512x64xbf16>) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c64 = arith.constant 64 : i64
    %c512 = arith.constant 512 : i64
    %qid = arith.constant 1 : i32
    %oid = arith.constant 2 : i32
    // Degenerate outer wrapper (trip count 1), exactly as observed in the real
    // lowered IR; the true lq_iters=2 iteration was unrolled into the two
    // launch_end blocks below.
    affine.for %i = 0 to 1 {
      // ---- unrolled launch iteration 0 ----
      affine.for %j = 0 to 2 {
        %a = airrt.dma_memcpy_nd(%qid, %c0, %c0, %q[%c0, %c0, %c0, %c0], [%c1, %c1, %c512, %c64], [%c0, %c0, %c64, %c1]) {metadata = @flashQKIn} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      }
      %b = airrt.dma_memcpy_nd(%oid, %c0, %c0, %o[%c0, %c0, %c0, %c0], [%c1, %c1, %c512, %c64], [%c0, %c0, %c64, %c1]) {metadata = @flashOut} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      airrt.wait_all %b {"air.launch_end"}
      // ---- unrolled launch iteration 1 ----
      affine.for %j = 0 to 2 {
        %a = airrt.dma_memcpy_nd(%qid, %c0, %c0, %q[%c0, %c0, %c0, %c0], [%c1, %c1, %c512, %c64], [%c0, %c0, %c64, %c1]) {metadata = @flashQKIn} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      }
      %c = airrt.dma_memcpy_nd(%oid, %c0, %c0, %o[%c0, %c0, %c0, %c0], [%c1, %c1, %c512, %c64], [%c0, %c0, %c64, %c1]) {metadata = @flashOut} : (i32, i64, i64, memref<512x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
      airrt.wait_all %c {"air.launch_end"}
    }
    %p = airrt.segment_load "flash_attn_seg" : i64
    return
  }
}
