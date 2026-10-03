//===- memtile_lock_race_fix_auto_blocked.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2 use-lock-race-condition-fix-auto=true" -verify-diagnostics

// Three buffers whose per-transfer locks would need 27 BDs in a 24-BD pool,
// none of which may take chain locks instead.

air.channel @w0 [1, 1]
air.channel @r0 [1, 1]
air.channel @r1 [1, 1]
air.channel @r2 [1, 1]
air.channel @r3 [1, 1]
air.channel @r4 [1, 1]
air.channel @r5 [1, 1]
func.func @fanouts_no_chain_lock() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %k0 = arith.constant 0 : index
      %k1 = arith.constant 1 : index
      %k2 = arith.constant 2 : index
      %k3 = arith.constant 3 : index
      %k4 = arith.constant 4 : index
      %k5 = arith.constant 5 : index
      %t_bufa, %bufa = air.execute -> (memref<6x8xbf16, 1>) {
        // expected-error@+1 {{a memtile BD pool needs 27 BDs and holds 24}}
        %alloc_bufa = memref.alloc() {air.no_chain_lock, air.no_split} : memref<6x8xbf16, 1>
        air.execute_terminator %alloc_bufa : memref<6x8xbf16, 1>
      }
      air.channel.get @w0[] (%bufa[] [] []) : (memref<6x8xbf16, 1>)
      air.channel.put @r0[] (%bufa[%k0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r1[] (%bufa[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r2[] (%bufa[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r3[] (%bufa[%k3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r4[] (%bufa[%k4, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r5[] (%bufa[%k5, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      %t_bufb, %bufb = air.execute -> (memref<6x8xbf16, 1>) {
        %alloc_bufb = memref.alloc() {air.no_chain_lock, air.no_split} : memref<6x8xbf16, 1>
        air.execute_terminator %alloc_bufb : memref<6x8xbf16, 1>
      }
      air.channel.get @w0[] (%bufb[] [] []) : (memref<6x8xbf16, 1>)
      air.channel.put @r0[] (%bufb[%k0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r1[] (%bufb[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r2[] (%bufb[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r3[] (%bufb[%k3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r4[] (%bufb[%k4, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r5[] (%bufb[%k5, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      %t_bufc, %bufc = air.execute -> (memref<6x8xbf16, 1>) {
        %alloc_bufc = memref.alloc() {air.no_chain_lock, air.no_split} : memref<6x8xbf16, 1>
        air.execute_terminator %alloc_bufc : memref<6x8xbf16, 1>
      }
      air.channel.get @w0[] (%bufc[] [] []) : (memref<6x8xbf16, 1>)
      air.channel.put @r0[] (%bufc[%k0, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r1[] (%bufc[%k1, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r2[] (%bufc[%k2, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r3[] (%bufc[%k3, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r4[] (%bufc[%k4, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.channel.put @r5[] (%bufc[%k5, %c0] [%c1_0, %c8] [%c8, %c1_0]) : (memref<6x8xbf16, 1>)
      air.herd @hw tile (%tx_hw, %ty_hw) in (%sx_hw=%c1_0, %sy_hw=%c1_0)
            attributes {x_loc = 6 : i64, y_loc = 2 : i64} {
        %tok, %l1 = air.execute -> (memref<48xbf16, 2>) {
          %aa = memref.alloc() : memref<48xbf16, 2>
          air.execute_terminator %aa : memref<48xbf16, 2>
        }
        air.channel.put @w0[] (%l1[] [] []) : (memref<48xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<48xbf16, 2>)
        air.channel.put @w0[] (%l1[] [] []) : (memref<48xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<48xbf16, 2>}
      }
      air.herd @h0 tile (%tx_h0, %ty_h0) in (%sx_h0=%c1_0, %sy_h0=%c1_0)
            attributes {x_loc = 0 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r0[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h1 tile (%tx_h1, %ty_h1) in (%sx_h1=%c1_0, %sy_h1=%c1_0)
            attributes {x_loc = 1 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r1[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h2 tile (%tx_h2, %ty_h2) in (%sx_h2=%c1_0, %sy_h2=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r2[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h3 tile (%tx_h3, %ty_h3) in (%sx_h3=%c1_0, %sy_h3=%c1_0)
            attributes {x_loc = 3 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r3[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h4 tile (%tx_h4, %ty_h4) in (%sx_h4=%c1_0, %sy_h4=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r4[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r4[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r4[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
      air.herd @h5 tile (%tx_h5, %ty_h5) in (%sx_h5=%c1_0, %sy_h5=%c1_0)
            attributes {x_loc = 5 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.get @r5[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r5[] (%l1[] [] []) : (memref<8xbf16, 2>)
        air.channel.get @r5[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %dl = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
