//===- opt_memtile_dma_bds_put_chains.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu2" -split-input-file | FileCheck %s

// Consecutive puts of one L2 buffer on one channel, at lattice offsets, fold
// into one put with the lattice as its outer dims.

// A core's six packets, two rounds of every other packet; the unit dims'
// offset moves to the innermost dim, which then splits to fit the BD's wrap.
// CHECK-LABEL: func.func @lattice_2d
// CHECK: air.channel.put async {{.*}}@chan[{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c5120{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c8{{.*}}, %c640{{.*}}] [%c20480{{.*}}, %c40960{{.*}}, %c640{{.*}}, %c1{{.*}}])
// CHECK-NOT: air.channel.put
module {
  air.channel @chan [1, 4]
  func.func @lattice_2d() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<1x6x4x5120xi16, 1 : i32>) {
          %a = memref.alloc() : memref<1x6x4x5120xi16, 1 : i32>
          air.execute_terminator %a : memref<1x6x4x5120xi16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 0, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 2, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 4, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p3 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 1, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p4 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 3, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p5 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 5, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2, %p3, %p4, %p5]
      }
    }
    return
  }
}

// -----

// Evenly spaced puts, each depending on the one before: one more dim.
// CHECK-LABEL: func.func @lattice_1d
// CHECK: air.channel.put async {{.*}}@chan[{{.*}}] (%{{.*}}[0, 4, 0] [3, 4, 8] [128, 16, 1])
// CHECK-NOT: air.channel.put
module {
  air.channel @chan [1, 4]
  func.func @lattice_1d() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<64x16xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<64x16xbf16, 1 : i32>
          air.execute_terminator %a : memref<64x16xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c2] (%buf[4, 0] [4, 8] [16, 1]) : (memref<64x16xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0, %p0] @chan[%c0, %c2] (%buf[12, 0] [4, 8] [16, 1]) : (memref<64x16xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0, %p1] @chan[%c0, %c2] (%buf[20, 0] [4, 8] [16, 1]) : (memref<64x16xbf16, 1 : i32>)
        %w = air.wait_all async [%p2]
      }
    }
    return
  }
}

// -----

// Descending offsets would need a negative stride: left as they are.
// CHECK-LABEL: func.func @descending
// CHECK-COUNT-3: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @descending() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %t0, %buf = air.execute -> (memref<3x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<3x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<3x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c3] (%buf[128] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0, %p0] @chan[%c0, %c3] (%buf[64] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0, %p1] @chan[%c0, %c3] (%buf[0] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p2]
      }
    }
    return
  }
}

// -----

// A put of another buffer in between ends the chain.
// CHECK-LABEL: func.func @other_buffer
// CHECK-COUNT-3: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @other_buffer() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %t0, %buf = air.execute -> (memref<3x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<3x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<3x64xbf16, 1 : i32>
        }
        %t1, %buf2 = air.execute -> (memref<3x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<3x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<3x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c3] (%buf[0] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t1] @chan[%c0, %c3] (%buf2[64] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c3] (%buf[128] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2]
      }
    }
    return
  }
}

// -----

// Offsets 0, 128, 64 do not form a lattice, nor does any run of them that
// starts a chain: the puts stay.
// CHECK-LABEL: func.func @not_a_lattice
// CHECK-COUNT-3: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @not_a_lattice(%dyn: index) {
    %0 = air.launch async () in () args(%d0=%dyn) : index {
      %1 = air.segment @seg async args(%d=%d0) : index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[128] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2]
      }
    }
    return
  }
}

// -----

// The second put waits on a token the first does not: folding would
// start it early. The puts stay.
// CHECK-LABEL: func.func @extra_dependency
// CHECK-COUNT-2: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @extra_dependency(%dyn: index) {
    %0 = air.launch async () in () args(%d0=%dyn) : index {
      %1 = air.segment @seg async args(%d=%d0) : index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %t1 = air.wait_all async
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0, %t1] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1]
      }
    }
    return
  }
}

// -----

// The first put's token is used before the chain ends: the folded put's
// token would not dominate that use. The puts stay.
// CHECK-LABEL: func.func @token_used_inside_chain
// CHECK-COUNT-2: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @token_used_inside_chain(%dyn: index) {
    %0 = air.launch async () in () args(%d0=%dyn) : index {
      %1 = air.segment @seg async args(%d=%d0) : index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %e = air.wait_all async [%p0]
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%e, %p1]
      }
    }
    return
  }
}

// -----

// The buffer is filled in two halves: its puts stay one per slot.
// CHECK-LABEL: func.func @filled_by_slots
// CHECK-COUNT-3: air.channel.put async {{.*}}@chan[%c0{{.*}}, %c1{{.*}}]
module {
  air.channel @chan [1, 4]
  func.func @filled_by_slots(%dyn: index) {
    %0 = air.launch async () in () args(%d0=%dyn) : index {
      %1 = air.segment @seg async args(%d=%d0) : index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %g0 = air.channel.get async [%t0] @chan[%c0, %c0] (%buf[0] [128] [1]) : (memref<4x64xbf16, 1 : i32>)
        %g1 = air.channel.get async [%g0] @chan[%c0, %c0] (%buf[128] [128] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p0 = air.channel.put async [%g1] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%g1] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%g1] @chan[%c0, %c1] (%buf[128] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2]
      }
    }
    return
  }
}

// -----

// A dynamic offset: the puts stay.
// CHECK-LABEL: func.func @dynamic_offset
// CHECK-COUNT-2: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @dynamic_offset(%dyn: index) {
    %0 = air.launch async () in () args(%d0=%dyn) : index {
      %1 = air.segment @seg async args(%d=%d0) : index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[%d] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1]
      }
    }
    return
  }
}

// -----

// Two chains on different channel indices, interleaved: each folds on its
// own.
// CHECK-LABEL: func.func @interleaved_chains
// CHECK-DAG: air.channel.put async {{.*}}@chan[%c0{{.*}}, %c1{{.*}}] (%{{.*}}[%c0{{.*}}] [%c192{{.*}}] [%c1{{.*}}])
// CHECK-DAG: air.channel.put async {{.*}}@chan[%c0{{.*}}, %c2{{.*}}] (%{{.*}}[%c32{{.*}}] [%c192{{.*}}] [%c1{{.*}}])
// CHECK-NOT: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @interleaved_chains(%dyn: index) {
    %0 = air.launch async () in () args(%d0=%dyn) : index {
      %1 = air.segment @seg async args(%d=%d0) : index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %q0 = air.channel.put async [%t0] @chan[%c0, %c2] (%buf[32] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %q1 = air.channel.put async [%t0] @chan[%c0, %c2] (%buf[96] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[128] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %q2 = air.channel.put async [%t0] @chan[%c0, %c2] (%buf[160] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2, %q0, %q1, %q2]
      }
    }
    return
  }
}

// -----

// Folding would give a put with more dims than a BD has: the puts stay.
// CHECK-LABEL: func.func @too_many_dims
// CHECK-COUNT-4: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @too_many_dims(%dyn: index) {
    %0 = air.launch async () in () args(%d0=%dyn) : index {
      %1 = air.segment @seg async args(%d=%d0) : index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<2x4x8x8xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<2x4x8x8xbf16, 1 : i32>
          air.execute_terminator %a : memref<2x4x8x8xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 0, 0, 0] [2, 2, 2, 8] [256, 64, 8, 1]) : (memref<2x4x8x8xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 2, 0, 0] [2, 2, 2, 8] [256, 64, 8, 1]) : (memref<2x4x8x8xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 0, 2, 0] [2, 2, 2, 8] [256, 64, 8, 1]) : (memref<2x4x8x8xbf16, 1 : i32>)
        %p3 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 2, 2, 0] [2, 2, 2, 8] [256, 64, 8, 1]) : (memref<2x4x8x8xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2, %p3]
      }
    }
    return
  }
}

// -----

// A re-fed buffer: its count is tied to its drain ops, so its puts stay.
// CHECK-LABEL: func.func @refed_buffer
// CHECK-COUNT-3: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @refed_buffer() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() {air.refeed_count = 2 : i32} : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[128] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2]
      }
    }
    return
  }
}

// -----

// A re-fed put is not folded; the two puts after it, neither re-fed, still
// fold.
// CHECK-LABEL: func.func @refed_put
// CHECK: air.channel.put async {{.*}} {air.refeed_count = 2 : i32}
// CHECK: air.channel.put async {{.*}}[%c64{{.*}}] [%c128{{.*}}] [%c1{{.*}}]
// CHECK-NOT: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @refed_put() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) {air.refeed_count = 2 : i32} : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[128] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2]
      }
    }
    return
  }
}

// -----

// A padded put sends extra zeros: the chain is not folded.
// CHECK-LABEL: func.func @padded_put
// CHECK-COUNT-2: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @padded_put() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) {pad_after = array<i32: 4>, pad_before = array<i32: 0>} : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1]
      }
    }
    return
  }
}

// -----

// A get between the puts: moving the first send past it could stall the
// consumer. The puts stay.
// CHECK-LABEL: func.func @get_in_between
// CHECK-COUNT-2: air.channel.put async
module {
  air.channel @chan [1, 4]
  air.channel @ack [1, 1]
  func.func @get_in_between() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %t1, %x = air.execute -> (memref<64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<64xbf16, 1 : i32>
          air.execute_terminator %a : memref<64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %g = air.channel.get async [%t1] @ack[] (%x[] [] []) : (memref<64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %g, %p1]
      }
    }
    return
  }
}

// -----

// A put on the same channel at an index that may equal the chain's: moving
// the first send past it could reorder that FIFO. The puts stay.
// CHECK-LABEL: func.func @same_channel_unknown_index
// CHECK-COUNT-3: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @same_channel_unknown_index(%arg0: index, %arg1: index) {
    %0 = air.launch async () in () args(%ai=%arg0, %aj=%arg1) : index, index {
      %1 = air.segment @seg async args(%i=%ai, %j=%aj) : index, index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %i] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %q = air.channel.put async [%t0] @chan[%c0, %j] (%buf[128] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %i] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %q, %p1]
      }
    }
    return
  }
}

// -----

// The buffer is filled through views, which may be slots: the puts stay.
// CHECK-LABEL: func.func @filled_through_views
// CHECK-COUNT-2: air.channel.put async {{.*}}@chan
module {
  air.channel @chan [1, 4]
  air.channel @in [1, 1]
  func.func @filled_through_views() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %s0 = memref.subview %buf[0, 0] [2, 64] [1, 1] : memref<4x64xbf16, 1 : i32> to memref<2x64xbf16, strided<[64, 1]>, 1 : i32>
        %s1 = memref.subview %buf[2, 0] [2, 64] [1, 1] : memref<4x64xbf16, 1 : i32> to memref<2x64xbf16, strided<[64, 1], offset: 128>, 1 : i32>
        %g0 = air.channel.get async [%t0] @in[] (%s0[] [] []) : (memref<2x64xbf16, strided<[64, 1]>, 1 : i32>)
        %g1 = air.channel.get async [%t0] @in[] (%s1[] [] []) : (memref<2x64xbf16, strided<[64, 1], offset: 128>, 1 : i32>)
        %p0 = air.channel.put async [%g0, %g1] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%g0, %g1] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1]
      }
    }
    return
  }
}

// -----

// An op after the chain waits only on the first put: tying it to the whole
// chain could stall the consumer of the second. The puts stay.
// CHECK-LABEL: func.func @early_token_user
// CHECK-COUNT-2: air.channel.put async {{.*}}@chan
module {
  air.channel @chan [1, 4]
  air.channel @ctl [1, 1]
  func.func @early_token_user() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<4x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<4x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<4x64xbf16, 1 : i32>
        }
        %t1, %x = air.execute -> (memref<64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<64xbf16, 1 : i32>
          air.execute_terminator %a : memref<64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[64] [64] [1]) : (memref<4x64xbf16, 1 : i32>)
        %ctl = air.channel.put async [%p0, %t1] @ctl[] (%x[] [] []) : (memref<64xbf16, 1 : i32>)
        %w = air.wait_all async [%ctl, %p1]
      }
    }
    return
  }
}

// -----

// A non-unit innermost stride: the innermost offset counts elements, so the
// puts at 0 and 8 are 8 elements apart, not 16, and together they read every
// other element of the first 16.
// CHECK-LABEL: func.func @inner_stride
// CHECK: air.channel.put async {{.*}}@chan[{{.*}}] (%{{.*}}[%c0{{.*}}] [%c8{{.*}}] [%c2{{.*}}])
// CHECK-NOT: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @inner_stride() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<32xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<32xbf16, 1 : i32>
          air.execute_terminator %a : memref<32xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0] [4] [2]) : (memref<32xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[8] [4] [2]) : (memref<32xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1]
      }
    }
    return
  }
}
