//===- air_shared_segment_name_to_one_device.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Segments that share a name are one device, even in different funcs. The
// device holds each distinct body once, and the shim DMAs of every func's
// launch: op_a and op_b each drive their own herd, and op_a_hi repeats op_a's
// body on other data.

// RUN: air-opt %s -air-to-aie='device=npu2' --split-input-file -verify-diagnostics | FileCheck %s
// RUN: air-opt %s -air-to-aie='device=npu2' --split-input-file -verify-diagnostics | FileCheck %s --check-prefix=MEMTILES

// The bodies share the device's memtiles: the second one adds none.
// MEMTILES: aie.device(npu2) @seg
// MEMTILES-COUNT-2: aie.logical_tile<MemTile>
// MEMTILES-NOT: aie.logical_tile<MemTile>
// MEMTILES: airrt.segment_metadata
// CHECK: aie.device(npu2) @seg
// CHECK-DAG: %[[T0:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[T1:.*]] = aie.tile(1, 2)
// CHECK-COUNT-2: aie.core(
// CHECK-NOT: aie.core(
// CHECK-DAG: aie.flow(%{{.*}}, DMA : {{[0-9]}}, %[[T0]], DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}, DMA : {{[0-9]}}, %[[T1]], DMA : 0)
// CHECK-DAG: aie.flow(%[[T0]], DMA : 0,
// CHECK-DAG: aie.flow(%[[T1]], DMA : 0,
// CHECK-DAG: aie.shim_dma_allocation @air_in_ha(
// CHECK-DAG: aie.shim_dma_allocation @air_out_ha(
// CHECK-DAG: aie.shim_dma_allocation @air_in_hb(
// CHECK-DAG: aie.shim_dma_allocation @air_out_hb(
// CHECK: }
// CHECK-NOT: aie.device
// CHECK: airrt.segment_metadata{{.*}}sym_name = "seg"
// CHECK-COUNT-2: airrt.herd_metadata
// CHECK-NOT: airrt.herd_metadata
// CHECK: func.func @op_a(
// CHECK: air.channel.put {{.*}}@in_ha{{.*}}metadataArray = [{base = "air_in_ha"
// CHECK: func.func @op_b(
// CHECK: air.channel.put {{.*}}@in_hb{{.*}}metadataArray = [{base = "air_in_hb"
// CHECK: func.func @op_a_hi(
// CHECK: air.channel.put {{.*}}@in_ha{{.*}}metadataArray = [{base = "air_in_ha"

module {
  air.channel @in_ha [1, 1]
  air.channel @out_ha [1, 1]
  air.channel @in_hb [1, 1]
  air.channel @out_hb [1, 1]

  func.func @op_a(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h_ha = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %t1, %b = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0, %t1] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %e = air.execute [%get] {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i64 = arith.constant 64 : index
            %k = arith.constant 2 : i32
            scf.for %i = %i0 to %i64 step %i1 {
              %v = memref.load %a[%i] : memref<64xi32, 2>
              %r = arith.addi %v, %k : i32
              memref.store %r, %b[%i] : memref<64xi32, 2>
            }
          }
          %put = air.channel.put async [%e] @out_ha[] (%b[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
          %d1 = air.execute [%put] {
            memref.dealloc %b : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }

  func.func @op_b(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_hb[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_hb[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h_hb = air.herd @hb async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 1 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %t1, %b = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0, %t1] @in_hb[] (%a[] [] []) : (memref<64xi32, 2>)
          %e = air.execute [%get] {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i64 = arith.constant 64 : index
            %k = arith.constant 3 : i32
            scf.for %i = %i0 to %i64 step %i1 {
              %v = memref.load %a[%i] : memref<64xi32, 2>
              %r = arith.muli %v, %k : i32
              memref.store %r, %b[%i] : memref<64xi32, 2>
            }
          }
          %put = air.channel.put async [%e] @out_hb[] (%b[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
          %d1 = air.execute [%put] {
            memref.dealloc %b : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }

  func.func @op_a_hi(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 64 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h_ha = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %t1, %b = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0, %t1] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %e = air.execute [%get] {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i64 = arith.constant 64 : index
            %k = arith.constant 2 : i32
            scf.for %i = %i0 to %i64 step %i1 {
              %v = memref.load %a[%i] : memref<64xi32, 2>
              %r = arith.addi %v, %k : i32
              memref.store %r, %b[%i] : memref<64xi32, 2>
            }
          }
          %put = air.channel.put async [%e] @out_ha[] (%b[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
          %d1 = air.execute [%put] {
            memref.dealloc %b : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// A herd may appear in only one distinct body of the device.

module {
  air.channel @in_ha [1, 1]
  air.channel @out_ha [1, 1]
  air.channel @in_hb [1, 1]
  air.channel @out_hb [1, 1]

  func.func @op_a(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      // expected-note@+1 {{the other body}}
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h_ha = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %t1, %b = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0, %t1] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %e = air.execute [%get] {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i64 = arith.constant 64 : index
            %k = arith.constant 2 : i32
            scf.for %i = %i0 to %i64 step %i1 {
              %v = memref.load %a[%i] : memref<64xi32, 2>
              %r = arith.addi %v, %k : i32
              memref.store %r, %b[%i] : memref<64xi32, 2>
            }
          }
          %put = air.channel.put async [%e] @out_ha[] (%b[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
          %d1 = air.execute [%put] {
            memref.dealloc %b : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }

  func.func @op_a4(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        // expected-error@+1 {{is also in another body of segment @seg, which differs from this one}}
        %h_ha = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %t1, %b = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0, %t1] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %e = air.execute [%get] {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i64 = arith.constant 64 : index
            %k = arith.constant 4 : i32
            scf.for %i = %i0 to %i64 step %i1 {
              %v = memref.load %a[%i] : memref<64xi32, 2>
              %r = arith.addi %v, %k : i32
              memref.store %r, %b[%i] : memref<64xi32, 2>
            }
          }
          %put = air.channel.put async [%e] @out_ha[] (%b[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
          %d1 = air.execute [%put] {
            memref.dealloc %b : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// Separate compilations of one body name their channels differently: op_c
// repeats op_a's body on channels of its own, so the device keeps op_a's, and
// op_c's launch is pointed at them.

// CHECK: aie.device(npu2) @seg
// CHECK-COUNT-1: aie.core(
// CHECK-NOT: aie.core(
// CHECK: func.func @op_a(
// CHECK: air.channel.put {{.*}}@in_ha
// CHECK: func.func @op_c(
// CHECK: air.channel.put {{.*}}@in_ha{{.*}}metadataArray = [{base = "air_in_ha"
// CHECK: air.channel.get {{.*}}@out_ha{{.*}}metadataArray = [{base = "air_out_ha"

module {
  air.channel @in_ha [1, 1]
  air.channel @out_ha [1, 1]
  air.channel @in_hc [1, 1]
  air.channel @out_hc [1, 1]

  func.func @op_a(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_ha[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_ha[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h_ha = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %t1, %b = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0, %t1] @in_ha[] (%a[] [] []) : (memref<64xi32, 2>)
          %e = air.execute [%get] {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i64 = arith.constant 64 : index
            %k = arith.constant 2 : i32
            scf.for %i = %i0 to %i64 step %i1 {
              %v = memref.load %a[%i] : memref<64xi32, 2>
              %r = arith.addi %v, %k : i32
              memref.store %r, %b[%i] : memref<64xi32, 2>
            }
          }
          %put = air.channel.put async [%e] @out_ha[] (%b[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
          %d1 = air.execute [%put] {
            memref.dealloc %b : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }


  func.func @op_c(%arg0: memref<128xi32>, %arg1: memref<128xi32>) {
    %l = air.launch async () in () args(%in=%arg0, %out=%arg1) : memref<128xi32>, memref<128xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cn = arith.constant 0 : index
      %p = air.channel.put async @in_hc[] (%in[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %g = air.channel.get async @out_hc[] (%out[%cn] [%c64] [%c1]) : (memref<128xi32>)
      %s = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 2 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index
        %h_ha = air.herd @ha async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %t1, %b = air.execute -> (memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m : memref<64xi32, 2>
          }
          %get = air.channel.get async [%t0, %t1] @in_hc[] (%a[] [] []) : (memref<64xi32, 2>)
          %e = air.execute [%get] {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i64 = arith.constant 64 : index
            %k = arith.constant 2 : i32
            scf.for %i = %i0 to %i64 step %i1 {
              %v = memref.load %a[%i] : memref<64xi32, 2>
              %r = arith.addi %v, %k : i32
              memref.store %r, %b[%i] : memref<64xi32, 2>
            }
          }
          %put = air.channel.put async [%e] @out_hc[] (%b[] [] []) : (memref<64xi32, 2>)
          %d0 = air.execute [%put] {
            memref.dealloc %a : memref<64xi32, 2>
          }
          %d1 = air.execute [%put] {
            memref.dealloc %b : memref<64xi32, 2>
          }
        }
      }
    }
    return
  }

}
