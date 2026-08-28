//===- dma_vs_channel_equivalence.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// A design written with air.dma_memcpy_nd naming its channel must generate the
// SAME hardware as the same design written with explicit air.channel.put/get.
//
// This is the load-bearing claim behind letting a front end use DMAs: the two
// spellings are not merely both valid, they are the same design. It is not
// obvious -- air-dma-to-channel hoists each DMA's external half into its own
// scf.parallel wrapper, so the IR reaching air-fuse-channels, air-split-l2-memref
// and ping-pong is structurally different from what a hand-written builder emits
// (AIRToAIEPass.cpp carries a workaround keyed on exactly that wrapper).
//
// The comparison is on the generated aie.device, not on the whole module: the
// residual air IR can legitimately differ. The L3->L2->L1 pair below is the
// case in point -- the DMA form makes the L2 buffer an explicit herd operand, so
// air-dependency draws an edge from the L2 fill to the herd that the channel
// form has no reason to draw. That edge is redundant (the herd's own gets
// already order it) and it changes nothing downstream, which is precisely what
// comparing the device rather than the module establishes.
//
// Inputs live in Inputs/ as PAIRS: equiv_<case>_dma.mlir and
// equiv_<case>_chan.mlir must describe the same design.
//
// NOT covered, deliberately: a re-feed. An n-trip loop around a DMA does NOT
// lower to the same design as an n-trip loop around a put -- the core acquires
// once instead of n times, because a DMA is one put and one get inseparably
// while a re-feed is decoupled by construction. See the plan notes; re-feed
// stays hand-written.

// RUN: air-opt %S/Inputs/equiv_ab_dma.mlir  --pass-pipeline='builtin.module(air-dependency,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | air-opt --mlir-print-local-scope > %t.ab.dma
// RUN: air-opt %S/Inputs/equiv_ab_chan.mlir --pass-pipeline='builtin.module(air-dependency,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | air-opt --mlir-print-local-scope > %t.ab.chan
// RUN: diff %t.ab.dma %t.ab.chan

// RUN: air-opt %S/Inputs/equiv_b2_dma.mlir  --pass-pipeline='builtin.module(air-dependency,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | air-opt --mlir-print-local-scope > %t.b2.dma
// RUN: air-opt %S/Inputs/equiv_b2_chan.mlir --pass-pipeline='builtin.module(air-dependency,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | air-opt --mlir-print-local-scope > %t.b2.chan
// RUN: diff %t.b2.dma %t.b2.chan

// RUN: air-opt %S/Inputs/equiv_cv_dma.mlir  --pass-pipeline='builtin.module(air-dependency,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse,air-enforce-channel-fifo-order)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | air-opt --mlir-print-local-scope > %t.cv.dma
// RUN: air-opt %S/Inputs/equiv_cv_chan.mlir --pass-pipeline='builtin.module(air-dependency,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse,air-enforce-channel-fifo-order)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | air-opt --mlir-print-local-scope > %t.cv.chan
// RUN: diff %t.cv.dma %t.cv.chan

// The L3->L2->L1 pair: compare the generated device only, for the reason above.
// RUN: air-opt %S/Inputs/equiv_l2_dma.mlir  --pass-pipeline='builtin.module(air-dependency,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | awk '/aie.device/,/^  }$/' > %t.l2.dma
// RUN: air-opt %S/Inputs/equiv_l2_chan.mlir --pass-pipeline='builtin.module(air-dependency,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | awk '/aie.device/,/^  }$/' > %t.l2.chan
// RUN: diff %t.l2.dma %t.l2.chan

// A broadcast feed. This pair needs air-broadcast-detection and
// air-specialize-dma-broadcast ahead of air-dma-to-channel, and compares the
// device only: the two forms reach the same fan-out by different routes (the
// DMA form derives the guard, the channel form is written with it).
// RUN: air-opt %S/Inputs/equiv_bc_dma.mlir  --pass-pipeline='builtin.module(air-dependency,air-broadcast-detection,air-specialize-dma-broadcast,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | awk '/aie.device/,/^  }$/' > %t.bc.dma
// RUN: air-opt %S/Inputs/equiv_bc_chan.mlir --pass-pipeline='builtin.module(air-dependency,air-broadcast-detection,air-specialize-dma-broadcast,air-dma-to-channel,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0},canonicalize,cse)' | air-opt --pass-pipeline='builtin.module(air-to-aie{emit-while-loop=true row-offset=2 col-offset=0 device=npu2})' | awk '/aie.device/,/^  }$/' > %t.bc.chan
// RUN: diff %t.bc.dma %t.bc.chan

// This file is a driver for the pairs in Inputs/; it holds no IR of its own.
