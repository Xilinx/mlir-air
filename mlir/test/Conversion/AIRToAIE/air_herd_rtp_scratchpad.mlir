//===- air_herd_rtp_scratchpad.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --split-input-file -air-to-aie="output-elf=true" | FileCheck %s --check-prefix=ELF
// RUN: air-opt %s --split-input-file -air-to-aie | FileCheck %s --check-prefix=RTP

// A herd's runtime scalar reaches the core through the RTP buffer, which
// AIRRtToNpu writes with a control-plane write32. A static TXN binary (the
// full-ELF output) cannot encode a write32 whose value is not known at compile
// time, so on that target the operand goes through an mlir-aie scratchpad
// parameter instead: declared at module scope, read by the core, written by
// the host per dispatch.
//
// This is gated because the two have different host contracts. An unwritten
// scratchpad parameter reads as zero rather than failing, so turning it on
// everywhere would leave existing designs running with a silently wrong value.

// -----

// The operand is the function's own argument: unknowable at compile time, so
// under output-elf it becomes a parameter and the core reads it directly. The
// core also opts out of the lowering's own sync preamble -- AIR's herd lock
// already orders the cores against the parameter write, and a second protocol
// would only give the two a way to disagree.

// ELF: aiex.scratchpad_parameter @__air_param_rtherd_0 : i32
// ELF-LABEL: aie.device
// ELF: aie.core
// ELF: aiex.read_scratchpad_parameter @__air_param_rtherd_0 : i32
// ELF-NOT: memref.load %{{.*}}[%c0] : memref<1xi32>
// ELF: emit_parameter_sync_preamble = false

// RTP-NOT: aiex.scratchpad_parameter
// RTP-LABEL: aie.device
// RTP: %[[BUF:.*]] = aie.buffer(%{{.*}}) {{{.*}}sym_name = "__air_herd_rtp_
// RTP: aie.core
// RTP: memref.load %[[BUF]][%c0] : memref<1xi32>
func.func @runtime_scalar(%L: i32) {
  %c1 = arith.constant 1 : index
  air.herd @rtherd tile(%tx, %ty) in (%sx = %c1, %sy = %c1) args(%a = %L) : i32 {
    %buf = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    %0 = memref.load %buf[%zero] : memref<1xi32, 2>
    %1 = arith.addi %0, %a : i32
    memref.store %1, %buf[%zero] : memref<1xi32, 2>
    air.herd_terminator
  }
  return
}

// -----

// A constant operand keeps the RTP path even under output-elf. "Not an
// arith.constant" would be too weak a test to rely on -- this pass runs long
// before the canonicalization that folds an index_cast of a constant -- so the
// classifier requires the operand to trace to a block argument. Getting this
// wrong moves a value to the scratchpad with nobody to write it.

// ELF-NOT: aiex.scratchpad_parameter
// ELF-LABEL: aie.device
// ELF: %[[BUF:.*]] = aie.buffer(%{{.*}}) {{{.*}}sym_name = "__air_herd_rtp_
// ELF: aie.core
// ELF: memref.load %[[BUF]][%c0] : memref<1xi32>
func.func @constant_scalar() {
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  air.herd @cherd tile(%tx, %ty) in (%sx = %c1, %sy = %c1) args(%a = %c42) : i32 {
    %buf = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    %0 = memref.load %buf[%zero] : memref<1xi32, 2>
    %1 = arith.addi %0, %a : i32
    memref.store %1, %buf[%zero] : memref<1xi32, 2>
    air.herd_terminator
  }
  return
}

// -----

// An induction variable is a block argument too, but it varies WITHIN a
// dispatch while the scratchpad holds one value per dispatch. Routing it there
// would freeze it at whatever the host last wrote. It stays on the RTP path,
// which is rewritten per iteration.

// ELF-NOT: aiex.scratchpad_parameter
// ELF-LABEL: aie.device
// ELF: %[[BUF:.*]] = aie.buffer(%{{.*}}) {{{.*}}sym_name = "__air_herd_rtp_
// ELF: aie.core
// ELF: memref.load %[[BUF]][%c0] : memref<1xi32>
func.func @loop_varying_scalar() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %iv = %c0 to %c4 step %c1 {
    %ivi = arith.index_cast %iv : index to i32
    air.herd @lherd tile(%tx, %ty) in (%sx = %c1, %sy = %c1) args(%a = %ivi) : i32 {
      %buf = memref.alloc() : memref<1xi32, 2>
      %zero = arith.constant 0 : index
      %0 = memref.load %buf[%zero] : memref<1xi32, 2>
      %1 = arith.addi %0, %a : i32
      memref.store %1, %buf[%zero] : memref<1xi32, 2>
      air.herd_terminator
    }
  }
  return
}

// -----

// Mixed operands: the runtime one moves to the scratchpad, the constant one
// stays. The slot numbering is deliberately NOT compacted -- a runtime operand
// still consumes its RTP word, it just goes unread -- so that this pass and
// AIRRtToNpu cannot drift on which slot is which. Here that shows as the
// constant keeping slot 1 in a 2-element buffer.

// ELF: aiex.scratchpad_parameter @__air_param_mherd_0 : i32
// ELF-LABEL: aie.device
// ELF: %[[BUF:.*]] = aie.buffer(%{{.*}}) {{{.*}}sym_name = "__air_herd_rtp_{{.*}}} : memref<2xi32>
// ELF: aie.core
// ELF-DAG: aiex.read_scratchpad_parameter @__air_param_mherd_0 : i32
// ELF-DAG: memref.load %[[BUF]][%c1] : memref<2xi32>
func.func @mixed_scalars(%L: i32) {
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : i32
  air.herd @mherd tile(%tx, %ty) in (%sx = %c1, %sy = %c1) args(%a = %L, %b = %c42) : i32, i32 {
    %buf = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    %0 = memref.load %buf[%zero] : memref<1xi32, 2>
    %1 = arith.addi %0, %a : i32
    %2 = arith.addi %1, %b : i32
    memref.store %2, %buf[%zero] : memref<1xi32, 2>
    air.herd_terminator
  }
  return
}
