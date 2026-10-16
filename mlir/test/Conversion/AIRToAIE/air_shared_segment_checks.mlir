//===- air_shared_segment_checks.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2' --split-input-file -verify-diagnostics | FileCheck %s

// A repeat may carry other lock choices on its buffers: the device lowers the
// first body's buffers only, so the second func still repeats it.
// CHECK-LABEL: aie.device(npu2) @seg
// CHECK-COUNT-1: aie.core(
// CHECK-NOT: aie.core(

module {
  air.channel @in_ha [1, 1]
  air.channel @out_ha [1, 1]
  air.channel @in_hb [1, 1]
  air.channel @out_hb [1, 1]
  func.func @op_a(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %put = air.channel.put async [%get] @out_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }

  func.func @op_a_chained(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() {air.chain_lock} : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %put = air.channel.put async [%get] @out_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// Distinct bodies with herds on the same tile.

module {
  air.channel @in_ha [1, 1]
  air.channel @out_ha [1, 1]
  air.channel @in_hb [1, 1]
  air.channel @out_hb [1, 1]
  func.func @op_a(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        // expected-note@+1 {{the other herd}}
        %h = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %put = air.channel.put async [%get] @out_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }

  func.func @op_b(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_hb[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_hb[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        // expected-error@+1 {{shares tiles with a herd in another body of segment @seg}}
        %h = air.herd @hb async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0] @in_hb[] (%a[] [] []) : (memref<64xi32, 2>)
          %put = air.channel.put async [%get] @out_hb[] (%a[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// A shared name with an unrolled segment is rejected wherever that segment
// comes.

module {
  air.channel @in_ha [1, 1]
  air.channel @out_ha [1, 1]
  air.channel @in_hb [1, 1]
  air.channel @out_hb [1, 1]
  func.func @op_a(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %c2 = arith.constant 2 : index
      // expected-error@+1 {{shares its name with another segment, so it must not be unrolled}}
      %s = air.segment @seg async unroll(%ux, %uy) in (%usx=%c2, %usy=%c1) attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %put = air.channel.put async [%get] @out_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }

  func.func @op_b(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_hb[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_hb[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h = air.herd @hb async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0] @in_hb[] (%a[] [] []) : (memref<64xi32, 2>)
          %put = air.channel.put async [%get] @out_hb[] (%a[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// The device's DMA allocations belong to the first body's air.dma_memcpy_nd
// ops; a repeat cannot use them.

module {
  func.func @op_a(%arg0: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0) : memref<128xi32> {
      %s = air.segment @seg async args(%sin=%in) : memref<128xi32> attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%hin=%sin) : memref<128xi32> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %d = air.dma_memcpy_nd async [%t0] (%a[] [] [], %hin[%c0] [%c64] [%c1]) : (memref<64xi32, 2>, memref<128xi32>)
          %d0 = air.execute [%d] {
            memref.dealloc %a : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }

  func.func @op_a_again(%arg0: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0) : memref<128xi32> {
      %s = air.segment @seg async args(%sin=%in) : memref<128xi32> attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%hin=%sin) : memref<128xi32> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          // expected-error@+1 {{is in a repeat of another body of segment @seg; a shared body needs channels}}
          %d = air.dma_memcpy_nd async [%t0] (%a[] [] [], %hin[%c0] [%c64] [%c1]) : (memref<64xi32, 2>, memref<128xi32>)
          %d0 = air.execute [%d] {
            memref.dealloc %a : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }
}
