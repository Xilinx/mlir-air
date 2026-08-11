//===- strict_option.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// An over-supplied channel is reported either way; `strict` decides whether it
// stops the build. The diagnostic itself stays a warning -- only the pass
// result changes.
//
// The second and third runs check the exit code rather than the diagnostics:
// -verify-diagnostics reports on the expectations alone and hides whether the
// pass signalled failure, which is the whole point of the option.

// RUN: air-opt %s -air-verify-refeed-balance -verify-diagnostics
// RUN: air-opt %s -air-verify-refeed-balance -o /dev/null
// RUN: not air-opt %s -air-verify-refeed-balance="strict=true" -o /dev/null

air.channel @over [1] {air.refeed_count = 4 : i32}

func.func @surplus_gates_only_under_strict() {
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  %dst = memref.alloc() : memref<64xbf16, 2 : i32>
  // expected-warning @+3 {{air.channel @over[0] is unbalanced}}
  // expected-note @+2 {{the balance closes at air.refeed_count = 1}}
  // expected-note @+1 {{producer: 64 tokens x refeed 4}}
  air.channel.put @over[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  // expected-note @+1 {{consumer: 64 tokens}}
  air.channel.get @over[] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
  return
}
