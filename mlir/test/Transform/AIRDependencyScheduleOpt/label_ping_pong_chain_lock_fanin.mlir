//===- label_ping_pong_chain_lock_fanin.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Under the v2 chain lock an L2 buffer with N writers and one reader is drained
// by a chain ordering them W0 -> ... -> W{N-1} -> R -> W0. A herd feeding it
// gains nothing from ping-pong -- the chain will not take round r+1 from stage
// k until stage k-1 commits -- while the extra packet parks on a switchbox
// arbiter the router may have shared with an earlier stage, deadlocking it.
//
// Gated on chain-lock-v2: without it the fan-in uses a counted lock that
// imposes no order.

// RUN: air-opt %s -air-label-scf-for-to-ping-pong="chain-lock-v2=true" | FileCheck %s --check-prefix=GUARD
// RUN: air-opt %s -air-label-scf-for-to-ping-pong | FileCheck %s --check-prefix=OFF

// =============================================================================
// Case 1: the herd's @outA put reaches @fanin -- two writers (@outA[0],
// @outA[1]), one reader (@drain). Declined with the flag, labeled without,
// which pins the guard as inert for designs that do not ask for the v2 lock.
// =============================================================================

// GUARD-LABEL: func.func @fanin_denies
// GUARD:       scf.for
// GUARD-NOT:   hoist_alloc
// GUARD-NOT:   } {unroll
// GUARD:       return

// OFF-LABEL: func.func @fanin_denies
// OFF:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// OFF:       } {unroll = 2 : i32}
// OFF:       return

module {
  air.channel @load_chan [1]
  air.channel @outA [2]
  air.channel @drain [1]
  air.channel @stageA [2]
  air.channel @stageB [2]
  air.channel @loneA [1]
  air.channel @loneDrain [1]
  air.channel @dmaStage [1]
  air.channel @dmaFan [2]
  air.channel @l1StageA [1]
  air.channel @l1StageB [2]

  func.func @fanin_denies() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        // The fan-in buffer: two writers, one reader.
        %tok_f, %fanin = air.execute -> (memref<64x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x32xbf16, 1>
          air.execute_terminator %alloc : memref<64x32xbf16, 1>
        }
        %w0 = air.channel.get async [%tok_f] @outA[%c0] (%fanin[%c0, %c0] [%c1_s, %c1_s] [%c1_s, %c1_s]) : (memref<64x32xbf16, 1>)
        %w1 = air.channel.get async [%tok_f] @outA[%c1_s] (%fanin[%c0, %c0] [%c1_s, %c1_s] [%c1_s, %c1_s]) : (memref<64x32xbf16, 1>)
        %r0 = air.channel.put async [%w0, %w1] @drain[] (%fanin[] [] []) : (memref<64x32xbf16, 1>)
        %2 = air.herd @herd_fanin async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %tok0 = air.wait_all async
          %3 = scf.for %arg10 = %c0_h to %c512 step %c64 iter_args(%arg11 = %tok0) -> (!air.async.token) {
            %tok_a, %buf = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%tok_a] @load_chan[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
            %tok_d = air.execute [%fill] {
              memref.dealloc %buf : memref<32x32xbf16, 2>
            }
            scf.yield %tok_d : !air.async.token
          }
          // The herd's output: what makes this herd a producer for the chain.
          %tok_o, %out = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %4 = air.channel.put async [%3, %tok_o] @outA[%arg21] (%out[] [] []) : (memref<32x32xbf16, 2>)
        }
      }
    }
    return
  }

// =============================================================================
// Case 2: the herd reaches @fanin only through a staging hop -- herd ->
// @stageA -> L2 -> @stageB -> fan-in. The shape gemma4's projection blocks
// have: a per-column memtile gathers four rows before forwarding one packet.
// So the walk must follow L2-to-L2 edges, not stop at the first buffer.
// =============================================================================

// GUARD-LABEL: func.func @staged_fanin_denies
// GUARD:       scf.for
// GUARD-NOT:   hoist_alloc
// GUARD-NOT:   } {unroll
// GUARD:       return

// OFF-LABEL: func.func @staged_fanin_denies
// OFF:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// OFF:       } {unroll = 2 : i32}
// OFF:       return

  func.func @staged_fanin_denies() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 2 : i32} {
      %1 = air.segment async {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        // Staging buffer: one writer, one reader -- not itself a chain.
        %tok_s, %stage = air.execute -> (memref<32x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 1>
          air.execute_terminator %alloc : memref<32x32xbf16, 1>
        }
        %s0 = air.channel.get async [%tok_s] @stageA[%c0] (%stage[] [] []) : (memref<32x32xbf16, 1>)
        %s1 = air.channel.put async [%s0] @stageB[%c0] (%stage[] [] []) : (memref<32x32xbf16, 1>)
        // The fan-in it feeds.
        %tok_f, %fanin = air.execute -> (memref<64x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x32xbf16, 1>
          air.execute_terminator %alloc : memref<64x32xbf16, 1>
        }
        %w0 = air.channel.get async [%tok_f] @stageB[%c0] (%fanin[%c0, %c0] [%c1_s, %c1_s] [%c1_s, %c1_s]) : (memref<64x32xbf16, 1>)
        %w1 = air.channel.get async [%tok_f] @stageB[%c1_s] (%fanin[%c0, %c0] [%c1_s, %c1_s] [%c1_s, %c1_s]) : (memref<64x32xbf16, 1>)
        %r0 = air.channel.put async [%w0, %w1] @drain[] (%fanin[] [] []) : (memref<64x32xbf16, 1>)
        %2 = air.herd @herd_staged async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0_h = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %tok0 = air.wait_all async
          %3 = scf.for %arg10 = %c0_h to %c512 step %c64 iter_args(%arg11 = %tok0) -> (!air.async.token) {
            %tok_a, %buf = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%tok_a] @load_chan[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
            %tok_d = air.execute [%fill] {
              memref.dealloc %buf : memref<32x32xbf16, 2>
            }
            scf.yield %tok_d : !air.async.token
          }
          %tok_o, %out = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %4 = air.channel.put async [%3, %tok_o] @stageA[%arg21] (%out[] [] []) : (memref<32x32xbf16, 2>)
        }
      }
    }
    return
  }

// =============================================================================
// Case 3 (POSITIVE control): same shape, but the output lands in a
// single-writer L2 buffer -- the legacy 1:1 lock, no imposed order to run ahead
// of, so ping-pong survives even with the guard on. Pins the guard against
// firing on any herd that merely writes L2.
// =============================================================================

// GUARD-LABEL: func.func @single_writer_labels
// GUARD:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// GUARD:       } {unroll = 2 : i32}
// GUARD:       return

// OFF-LABEL: func.func @single_writer_labels
// OFF:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// OFF:       } {unroll = 2 : i32}
// OFF:       return

  func.func @single_writer_labels() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 3 : i32} {
      %1 = air.segment async {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %tok_f, %lone = air.execute -> (memref<32x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 1>
          air.execute_terminator %alloc : memref<32x32xbf16, 1>
        }
        %w0 = air.channel.get async [%tok_f] @loneA[%c0] (%lone[] [] []) : (memref<32x32xbf16, 1>)
        %r0 = air.channel.put async [%w0] @loneDrain[] (%lone[] [] []) : (memref<32x32xbf16, 1>)
        %2 = air.herd @herd_lone async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0_h = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %tok0 = air.wait_all async
          %3 = scf.for %arg10 = %c0_h to %c512 step %c64 iter_args(%arg11 = %tok0) -> (!air.async.token) {
            %tok_a, %buf = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%tok_a] @load_chan[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
            %tok_d = air.execute [%fill] {
              memref.dealloc %buf : memref<32x32xbf16, 2>
            }
            scf.yield %tok_d : !air.async.token
          }
          %tok_o, %out = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %4 = air.channel.put async [%3, %tok_o] @loneA[%c0_h] (%out[] [] []) : (memref<32x32xbf16, 2>)
        }
      }
    }
    return
  }

// =============================================================================
// Case 4: one writer is an air.dma_memcpy_nd, not a channel.get. AIRToAIE
// counts every air::MemcpyInterface endpoint and still builds the chain;
// counting only channel ops here would see one writer, call it 1:1, and leave
// the run-ahead on. Must still deny.
// =============================================================================

// GUARD-LABEL: func.func @dma_writer_denies
// GUARD:       scf.for
// GUARD-NOT:   hoist_alloc
// GUARD-NOT:   } {unroll
// GUARD:       return

// OFF-LABEL: func.func @dma_writer_denies
// OFF:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// OFF:       } {unroll = 2 : i32}
// OFF:       return

  func.func @dma_writer_denies(%arg0: memref<64x32xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) args(%arg7=%arg0) : memref<64x32xbf16> attributes {id = 4 : i32} {
      %1 = air.segment async args(%arg8=%arg7) : memref<64x32xbf16> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %tok_f, %fanin = air.execute -> (memref<64x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x32xbf16, 1>
          air.execute_terminator %alloc : memref<64x32xbf16, 1>
        }
        // Writer 1: a channel get fed by the herd.
        %w0 = air.channel.get async [%tok_f] @dmaFan[%c0] (%fanin[%c0, %c0] [%c1_s, %c1_s] [%c1_s, %c1_s]) : (memref<64x32xbf16, 1>)
        // Writer 2: a legacy dma_memcpy_nd. Not an air.channel op.
        %w1 = air.dma_memcpy_nd async [%tok_f] (%fanin[%c32, %c0] [%c32, %c32] [%c32, %c1_s], %arg8[%c0, %c0] [%c32, %c32] [%c32, %c1_s]) : (memref<64x32xbf16, 1>, memref<64x32xbf16>)
        %r0 = air.channel.put async [%w0, %w1] @drain[] (%fanin[] [] []) : (memref<64x32xbf16, 1>)
        %2 = air.herd @herd_dma async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0_h = arith.constant 0 : index
          %c64_h = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %tok0 = air.wait_all async
          %3 = scf.for %arg10 = %c0_h to %c512 step %c64_h iter_args(%arg11 = %tok0) -> (!air.async.token) {
            %tok_a, %buf = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%tok_a] @load_chan[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
            %tok_d = air.execute [%fill] {
              memref.dealloc %buf : memref<32x32xbf16, 2>
            }
            scf.yield %tok_d : !air.async.token
          }
          %tok_o, %out = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %4 = air.channel.put async [%3, %tok_o] @dmaFan[%c0_h] (%out[] [] []) : (memref<32x32xbf16, 2>)
        }
      }
    }
    return
  }

// =============================================================================
// Case 5: the staging hop is L1, not L2. AIR permits cross-space channel hops,
// so this is still a path into the chain; stopping at the first non-L2
// destination would miss it.
// =============================================================================

// GUARD-LABEL: func.func @l1_staged_fanin_denies
// GUARD:       scf.for
// GUARD-NOT:   hoist_alloc
// GUARD-NOT:   } {unroll
// GUARD:       return

// OFF-LABEL: func.func @l1_staged_fanin_denies
// OFF:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// OFF:       } {unroll = 2 : i32}
// OFF:       return

  func.func @l1_staged_fanin_denies() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 5 : i32} {
      %1 = air.segment async {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        // Staging buffer in L1 (space 2), reached from the herd and forwarded on.
        %tok_s, %stage = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %s0 = air.channel.get async [%tok_s] @l1StageA[%c0] (%stage[] [] []) : (memref<32x32xbf16, 2>)
        %s1 = air.channel.put async [%s0] @l1StageB[%c0] (%stage[] [] []) : (memref<32x32xbf16, 2>)
        %tok_f, %fanin = air.execute -> (memref<64x32xbf16, 1>) {
          %alloc = memref.alloc() : memref<64x32xbf16, 1>
          air.execute_terminator %alloc : memref<64x32xbf16, 1>
        }
        %w0 = air.channel.get async [%tok_f] @l1StageB[%c0] (%fanin[%c0, %c0] [%c1_s, %c1_s] [%c1_s, %c1_s]) : (memref<64x32xbf16, 1>)
        %w1 = air.channel.get async [%tok_f] @l1StageB[%c1_s] (%fanin[%c0, %c0] [%c1_s, %c1_s] [%c1_s, %c1_s]) : (memref<64x32xbf16, 1>)
        %r0 = air.channel.put async [%w0, %w1] @drain[] (%fanin[] [] []) : (memref<64x32xbf16, 1>)
        %2 = air.herd @herd_l1staged async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0_h = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %tok0 = air.wait_all async
          %3 = scf.for %arg10 = %c0_h to %c512 step %c64 iter_args(%arg11 = %tok0) -> (!air.async.token) {
            %tok_a, %buf = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%tok_a] @load_chan[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
            %tok_d = air.execute [%fill] {
              memref.dealloc %buf : memref<32x32xbf16, 2>
            }
            scf.yield %tok_d : !air.async.token
          }
          %tok_o, %out = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %4 = air.channel.put async [%3, %tok_o] @l1StageA[%c0_h] (%out[] [] []) : (memref<32x32xbf16, 2>)
        }
      }
    }
    return
  }
}
