//===- canonicalize_hierarchy_channel_dep.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize -split-input-file %s | FileCheck %s

// A hierarchy op uses every channel its body uses. The symbol never appears in
// air.herd's or air.segment's own attributes -- only on the channel ops nested
// inside -- so a dependence carried by that channel is invisible unless the
// region is walked.
//
// The alias test that gates dead-edge removal only runs when both ends touch a
// memref, which is why the hierarchy op below has to hold one that its body
// actually uses: an unused kernel operand is pruned first and the op stops
// touching memrefs at all. Given a producer staging through its own buffer and
// a consumer holding an unrelated one, the two clear the gate, share no memref,
// and the edge is dropped. The result still verifies and still looks scheduled;
// it has only had the channel's ordering constraint removed, which surfaces
// much later as a deadlocked or reordered run.

// CHECK-LABEL: func.func @herd_channel_dep_preserved
// CHECK: %[[PUT:[a-zA-Z0-9_]+]] = air.channel.put async {{.*}}@ch_h
// CHECK: air.herd @h async [{{.*}}%[[PUT]]]

module {
  air.channel @ch_h [1]
  func.func @herd_channel_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        // Staging buffer: what the producer sends. Unrelated to the herd's arg.
        %tk_stage, %stage = air.execute -> (memref<32xi16, 1 : i32>) {
          %a = memref.alloc() : memref<32xi16, 1 : i32>
          air.execute_terminator %a : memref<32xi16, 1 : i32>
        }
        // Persistent buffer the herd carries as a kernel operand, and uses.
        %tk_keep, %keep = air.execute -> (memref<64xi16, 2 : i32>) {
          %a = memref.alloc() : memref<64xi16, 2 : i32>
          air.execute_terminator %a : memref<64xi16, 2 : i32>
        }
        %p = air.channel.put async [%tk_stage] @ch_h[%c0_s] (%stage[] [] []) : (memref<32xi16, 1 : i32>)
        // The herd may not start before the put: its body is the matching get.
        // The two share no memref, so only the channel symbol carries it.
        %h = air.herd @h async [%p, %tk_keep] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argkeep=%keep) : memref<64xi16, 2 : i32> {
          %c0_h = arith.constant 0 : index
          %c0_i16 = arith.constant 0 : i16
          %tk_in, %in = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %g = air.channel.get async [%tk_in] @ch_h[%c0_h] (%in[] [] []) : (memref<32xi16, 2 : i32>)
          %w = air.execute [%g] {
            %v = memref.load %in[%c0_h] : memref<32xi16, 2 : i32>
            memref.store %v, %argkeep[%c0_h] : memref<64xi16, 2 : i32>
          }
        }
      }
    }
    return
  }
}

// -----

// Same for air.segment, which is equally IsolatedFromAbove and equally opaque
// from the outside.

// CHECK-LABEL: func.func @segment_channel_dep_preserved
// CHECK: %[[PUT:[a-zA-Z0-9_]+]] = air.channel.put async {{.*}}@ch_s
// CHECK: air.segment @seg2 async [{{.*}}%[[PUT]]]

module {
  air.channel @ch_s [1]
  func.func @segment_channel_dep_preserved(%ext: memref<64xi16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%larg=%ext) : memref<64xi16> {
      %c0_o = arith.constant 0 : index
      %tk_stage, %stage = air.execute -> (memref<32xi16>) {
        %a = memref.alloc() : memref<32xi16>
        air.execute_terminator %a : memref<32xi16>
      }
      %p = air.channel.put async [%tk_stage] @ch_s[%c0_o] (%stage[] [] []) : (memref<32xi16>)
      %s = air.segment @seg2 async [%p] args(%skeep=%larg) : memref<64xi16> {
        %c0_s = arith.constant 0 : index
        %tk_in, %in = air.execute -> (memref<32xi16, 1 : i32>) {
          %a = memref.alloc() : memref<32xi16, 1 : i32>
          air.execute_terminator %a : memref<32xi16, 1 : i32>
        }
        %g = air.channel.get async [%tk_in] @ch_s[%c0_s] (%in[] [] []) : (memref<32xi16, 1 : i32>)
        %w = air.execute [%g] {
          %v = memref.load %in[%c0_s] : memref<32xi16, 1 : i32>
          memref.store %v, %skeep[%c0_s] : memref<64xi16>
        }
      }
    }
    return
  }
}

// -----

// air.execute is async and holds a region too, so it has the same blind spot.

// CHECK-LABEL: func.func @execute_channel_dep_preserved
// CHECK: %[[PUT:[a-zA-Z0-9_]+]] = air.channel.put async {{.*}}@ch_e
// CHECK: air.execute [{{.*}}%[[PUT]]{{.*}}]

module {
  air.channel @ch_e [1]
  func.func @execute_channel_dep_preserved(%ext: memref<64xi16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%larg=%ext) : memref<64xi16> {
      %c0_o = arith.constant 0 : index
      %tk_stage, %stage = air.execute -> (memref<32xi16>) {
        %a = memref.alloc() : memref<32xi16>
        air.execute_terminator %a : memref<32xi16>
      }
      %p = air.channel.put async [%tk_stage] @ch_e[%c0_o] (%stage[] [] []) : (memref<32xi16>)
      %tk_in, %in = air.execute -> (memref<32xi16>) {
        %a = memref.alloc() : memref<32xi16>
        air.execute_terminator %a : memref<32xi16>
      }
      %e = air.execute [%p, %tk_in] {
        air.channel.get @ch_e[%c0_o] (%in[] [] []) : (memref<32xi16>)
      }
    }
    return
  }
}

// -----

// The other direction, which is what stops this from being a blanket "never
// prune an edge into a hierarchy op". Walking the regions makes the analysis
// see more symbols, so it could start preserving edges that genuinely are
// dead. Here the two herds touch different channels and share no memref: the
// symbol sets are disjoint even after the walk, and the edge still goes.

// CHECK-LABEL: func.func @independent_hierarchies_still_pruned
// A single token in the dependence list -- no comma -- so the edge from @ha
// was pruned, as it should be.
// CHECK: air.herd @hb async [%{{[a-zA-Z0-9_]+}}] tile

module {
  air.channel @ch_a [1]
  air.channel @ch_b [1]
  func.func @independent_hierarchies_still_pruned() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %c1_s = arith.constant 1 : index
      %tk_a, %ka = air.execute -> (memref<64xi16, 2 : i32>) {
        %a = memref.alloc() : memref<64xi16, 2 : i32>
        air.execute_terminator %a : memref<64xi16, 2 : i32>
      }
      %tk_b, %kb = air.execute -> (memref<64xi16, 2 : i32>) {
        %a = memref.alloc() : memref<64xi16, 2 : i32>
        air.execute_terminator %a : memref<64xi16, 2 : i32>
      }
      %ha = air.herd @ha async [%tk_a] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%aa=%ka) : memref<64xi16, 2 : i32> {
        %c0_h = arith.constant 0 : index
        %tk_i, %i = air.execute -> (memref<32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32xi16, 2 : i32>
          air.execute_terminator %a : memref<32xi16, 2 : i32>
        }
        %g = air.channel.get async [%tk_i] @ch_a[%c0_h] (%i[] [] []) : (memref<32xi16, 2 : i32>)
        %w = air.execute [%g] {
          %v = memref.load %i[%c0_h] : memref<32xi16, 2 : i32>
          memref.store %v, %aa[%c0_h] : memref<64xi16, 2 : i32>
        }
      }
      %hb = air.herd @hb async [%ha, %tk_b] tile (%tx2, %ty2) in (%sx2=%c1_s, %sy2=%c1_s) args(%bb=%kb) : memref<64xi16, 2 : i32> {
        %c0_h2 = arith.constant 0 : index
        %tk_j, %j = air.execute -> (memref<32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32xi16, 2 : i32>
          air.execute_terminator %a : memref<32xi16, 2 : i32>
        }
        %g2 = air.channel.get async [%tk_j] @ch_b[%c0_h2] (%j[] [] []) : (memref<32xi16, 2 : i32>)
        %w2 = air.execute [%g2] {
          %v = memref.load %j[%c0_h2] : memref<32xi16, 2 : i32>
          memref.store %v, %bb[%c0_h2] : memref<64xi16, 2 : i32>
        }
      }
    }
    return
  }
}
