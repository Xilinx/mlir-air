//===- mixed_channel_bundle_indices.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A channel bundle is keyed by index only when every user's indices resolve to
// constants; otherwise all of them fall back to the symbol alone. That fallback
// is right when nothing could be told apart in the first place -- a bundle
// indexed throughout by a herd tile id, say -- because dataflow then orders the
// pairings anyway.
//
// It is wrong when the users *disagree*. Demoting a bundle because one user is
// unresolvable also strips the identity of the users that were perfectly
// resolvable, and two entries the runner could have distinguished are merged.
// The failure is remote from the cause: a pipeline seeded at entry [0] and
// drained at entry [N] has a drain that becomes ready as soon as its own seed
// completes, so it consumes the seed, the first real stage starves, and the run
// stalls hundreds of cycles later with nothing pointing at the channel.
//
// So the mixture is diagnosed, and uniformity -- all constant, or all dynamic --
// is left alone.

// RUN: air-runner %s -f mixed -m %S/custom_op/arch.json 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: air-runner %s -f all_dynamic -m %S/custom_op/arch.json 2>&1 | FileCheck %s --check-prefix=DYN
// RUN: air-runner %s -f all_constant -m %S/custom_op/arch.json 2>&1 | FileCheck %s --check-prefix=CONST

module {
  air.channel @mixed_io [4]
  air.channel @dyn_io [4]
  air.channel @const_io [2]

  // ---- 1. Mixed: constant seed/drain, loop-variable body. ----
  // This is the shape a rolled pipeline has, where the stage index selects the
  // link. The index reaches the body through an air.segment kernel operand,
  // because air.segment is IsolatedFromAbove -- so the check has to see through
  // that binding, which is also how a resolvable constant reaches a body after
  // unrolling.
  //
  // MIXED: op channel bundle @mixed_io is indexed both by constants and by
  // MIXED-SAME: values the runner cannot resolve
  // MIXED-SAME: unrolling the loop
  //
  // One diagnostic only: the merge is a property of the bundle, and the launch
  // and segment runner nodes must not each report it.
  // MIXED-NOT: op channel bundle @mixed_io is indexed both by constants
  func.func @mixed(%arg0: memref<64xi8>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%la=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %c0_o = arith.constant 0 : index
      %c1_o = arith.constant 1 : index
      %c3_o = arith.constant 3 : index
      %init = air.wait_all async
      %loop = scf.for %i = %c0_o to %c3_o step %c1_o iter_args(%d = %init) -> (!air.async.token) {
        %next = arith.addi %i, %c1_o : index
        %seg = air.segment async [%d] args(%si=%i, %sn=%next) : index, index attributes {id = 10 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
          %tok, %buf = air.execute -> (memref<64xi8, 1>) {
            %a = memref.alloc() : memref<64xi8, 1>
            air.execute_terminator %a : memref<64xi8, 1>
          }
          %rx = air.channel.get async [%tok] @mixed_io[%si] (%buf[] [] []) {id = 20 : i32} : (memref<64xi8, 1>)
          %tx = air.channel.put async [%rx] @mixed_io[%sn] (%buf[] [] []) {id = 21 : i32} : (memref<64xi8, 1>)
        }
        scf.yield %seg : !air.async.token
      }
      // The two constant-indexed users whose identity the fallback destroys.
      %in = air.channel.put async @mixed_io[%c0_o] (%la[] [] []) {id = 30 : i32} : (memref<64xi8>)
      %out = air.channel.get async [%in] @mixed_io[%c3_o] (%la[] [] []) {id = 31 : i32} : (memref<64xi8>)
    }
    return
  }

  // ---- 2. Uniformly dynamic: merged, and that is correct. ----
  // Every non-constant channel index in the rest of the runner corpus is of
  // this kind -- a herd tile id. Diagnosing it would fire on all of them, and
  // there is nothing wrong: the entries are the herd instances, dispatched as
  // one event, and merging is the intended semantics. Must stay silent and must
  // still reach the terminator.
  // DYN-NOT: is indexed both by constants
  // DYN: "name": "LaunchTerminator",
  // DYN: "ph": "E",
  func.func @all_dynamic(%arg0: memref<64xi8>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%la=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %seg = air.segment async attributes {id = 10 : i32, x_loc = 0 : i64, x_size = 2 : i64, y_loc = 0 : i64, y_size = 2 : i64} {
        %c2_s = arith.constant 2 : index
        %h = air.herd @h async tile (%tx, %ty) in (%tsx=%c2_s, %tsy=%c2_s) attributes {id = 100 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %tok, %buf = air.execute -> (memref<64xi8, 2>) {
            %a = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %a : memref<64xi8, 2>
          }
          %p = air.channel.put async [%tok] @dyn_io[%tx] (%buf[] [] []) {id = 20 : i32} : (memref<64xi8, 2>)
          %g = air.channel.get async [%p] @dyn_io[%ty] (%buf[] [] []) {id = 21 : i32} : (memref<64xi8, 2>)
        }
      }
    }
    return
  }

  // ---- 3. Uniformly constant: entries told apart, no diagnostic. ----
  // The same seed-and-drain shape as case 1, but with the stage index a real
  // constant one binding away -- which is what case 1 becomes after unrolling.
  // It must run: if these entries were merged, the drain would consume the seed
  // and the segment would starve, so reaching the terminator is the assertion
  // that constant indices still key the bundle.
  // CONST-NOT: is indexed both by constants
  // CONST: "name": "LaunchTerminator",
  // CONST: "ph": "E",
  func.func @all_constant(%arg0: memref<64xi8>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%la=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %c0_o = arith.constant 0 : index
      %c1_o = arith.constant 1 : index
      %seg = air.segment async args(%si=%c0_o, %sn=%c1_o) : index, index attributes {id = 10 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %tok, %buf = air.execute -> (memref<64xi8, 1>) {
          %a = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %a : memref<64xi8, 1>
        }
        %rx = air.channel.get async [%tok] @const_io[%si] (%buf[] [] []) {id = 20 : i32} : (memref<64xi8, 1>)
        %tx = air.channel.put async [%rx] @const_io[%sn] (%buf[] [] []) {id = 21 : i32} : (memref<64xi8, 1>)
      }
      %in = air.channel.put async @const_io[%c0_o] (%la[] [] []) {id = 30 : i32} : (memref<64xi8>)
      %out = air.channel.get async [%in] @const_io[%c1_o] (%la[] [] []) {id = 31 : i32} : (memref<64xi8>)
    }
    return
  }
}
