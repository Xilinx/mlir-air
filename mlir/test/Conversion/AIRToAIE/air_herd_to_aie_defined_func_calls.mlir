//===- air_herd_to_aie_defined_func_calls.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' --split-input-file | FileCheck %s

// llvm.noalias holds for every call: the first call passes two buffers, the
// second passes one buffer as both arguments, and the function writes one of
// them, so neither argument keeps it.
// CHECK-LABEL: aie.device
// CHECK: call @tile_body
// CHECK: call @tile_body
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32>, %{{.*}}: memref<64xi32>) {
module {
  func.func @two_calls() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %in = memref.alloc() : memref<64xi32, 2>
          %out = memref.alloc() : memref<64xi32, 2>
          func.call @tile_body(%in, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
          func.call @tile_body(%out, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}

// -----

// A defined function the cloned one calls is cloned in the same, default
// memory space, so the nested call stays well typed. It gets no llvm.noalias:
// its arguments are the caller's, which nothing proves disjoint.
// CHECK-LABEL: aie.device
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32> {llvm.noalias}, %[[B:.*]]: memref<64xi32> {llvm.noalias})
// CHECK: call @store_tile(%{{.*}}, %[[B]]) : (vector<16xi32>, memref<64xi32>) -> ()
// CHECK: func.func private @store_tile(%{{.*}}: vector<16xi32>, %{{.*}}: memref<64xi32>) {
// On the host both keep only a declaration.
// CHECK-LABEL: func.func @nested()
// CHECK: func.func private @tile_body(memref<64xi32, 2>, memref<64xi32, 2>){{$}}
// CHECK: func.func private @store_tile(vector<16xi32>, memref<64xi32, 2>){{$}}
module {
  func.func @nested() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %in = memref.alloc() : memref<64xi32, 2>
          %out = memref.alloc() : memref<64xi32, 2>
          func.call @tile_body(%in, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    func.call @store_tile(%v, %b) : (vector<16xi32>, memref<64xi32, 2>) -> ()
    return
  }
  func.func private @store_tile(%v: vector<16xi32>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}


// -----

// The async form the AIR pipeline produces: each buffer is an air.execute
// holding only its allocation, which is still distinct storage, so both
// arguments keep llvm.noalias.
// CHECK-LABEL: aie.device
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32> {llvm.noalias}, %{{.*}}: memref<64xi32> {llvm.noalias})
module {
  func.func @async_allocs() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      %1 = air.segment @seg async {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @h async tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %in = air.execute -> (memref<64xi32, 2>) {
            %a = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %a : memref<64xi32, 2>
          }
          %t1, %out = air.execute -> (memref<64xi32, 2>) {
            %a = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %a : memref<64xi32, 2>
          }
          %t2 = air.execute [%t0, %t1] {
            func.call @tile_body(%in, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
          }
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}

// -----

// A cloned callee that returns a memref: the call takes the callee's
// default-memory-space result type, cast back for its users.
// CHECK-LABEL: aie.device
// CHECK: %[[R:.*]] = func.call @pick(%{{.*}}) : (memref<64xi32>) -> memref<64xi32>
// CHECK: memref.memory_space_cast %[[R]] : memref<64xi32> to memref<64xi32, 2>
// CHECK: func.func private @pick(%{{.*}}: memref<64xi32>{{.*}}) -> memref<64xi32>
module {
  func.func @returns_memref() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %in = memref.alloc() : memref<64xi32, 2>
          %r = func.call @pick(%in) : (memref<64xi32, 2>) -> memref<64xi32, 2>
          %c0 = arith.constant 0 : index
          %z = arith.constant 0 : i32
          memref.store %z, %r[%c0] : memref<64xi32, 2>
        }
      }
    }
    return
  }
  func.func private @pick(%a: memref<64xi32, 2>) -> memref<64xi32, 2> {
    return %a : memref<64xi32, 2>
  }
}

// -----

// One air.execute yielding the same allocation twice: two results, one
// buffer, so the written argument gets no llvm.noalias.
// CHECK-LABEL: aie.device
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32>, %{{.*}}: memref<64xi32>) {
module {
  func.func @same_alloc_twice() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      %1 = air.segment @seg async {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @h async tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %a, %b = air.execute -> (memref<64xi32, 2>, memref<64xi32, 2>) {
            %m = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %m, %m : memref<64xi32, 2>, memref<64xi32, 2>
          }
          %t1 = air.execute [%t0] {
            func.call @tile_body(%a, %b) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
          }
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}

// -----

// An external function the cloned one calls is declared in the device as for
// a call from the core: layout dropped, the core's link_with, and the C
// interface.
// CHECK-LABEL: aie.device
// CHECK: func.func private @tile_body
// CHECK: %[[C:.*]] = memref.cast %{{.*}} : memref<16xi32, strided<[1], offset: ?>> to memref<16xi32>
// CHECK: call @ext_kernel(%[[C]]) : (memref<16xi32>) -> ()
// CHECK: func.func private @ext_kernel(memref<16xi32>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
module {
  func.func @nested_external() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {link_with = "kernel.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<64xi32, 2>
          func.call @tile_body(%buf, %tx) : (memref<64xi32, 2>, index) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %i: index) {
    %v = memref.subview %a[%i] [16] [1] : memref<64xi32, 2> to memref<16xi32, strided<[1], offset: ?>, 2>
    func.call @ext_kernel(%v) : (memref<16xi32, strided<[1], offset: ?>, 2>) -> ()
    return
  }
  func.func private @ext_kernel(memref<16xi32, strided<[1], offset: ?>, 2>)
}

// -----

// A function the host also calls keeps its body there.
// CHECK-LABEL: aie.device
// CHECK: func.func private @tile_body
// CHECK-LABEL: func.func @host_and_herd()
// CHECK: call @tile_body
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32>) {
// CHECK: vector.transfer_write
module {
  func.func @host_and_herd() {
    %c1 = arith.constant 1 : index
    %host = memref.alloc() : memref<64xi32>
    func.call @tile_body(%host) : (memref<64xi32>) -> ()
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) args(%lh=%host) : memref<64xi32> {
      air.segment @seg args(%sh=%lh) : memref<64xi32> {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) args(%b=%sh) : memref<64xi32> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          func.call @tile_body(%b) : (memref<64xi32>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %z = arith.constant dense<0> : vector<16xi32>
    vector.transfer_write %z, %a[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32>
    return
  }
}

// -----

// transform.loop.outline makes public functions. One only herds call is
// handled as a private one: the host keeps a private declaration.
// CHECK-LABEL: aie.device
// CHECK: func.func private @outlined
// CHECK-LABEL: func.func @public_callee()
// CHECK: func.func private @outlined(memref<64xi32, 2>){{$}}
module {
  func.func @public_callee() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<64xi32, 2>
          func.call @outlined(%buf) : (memref<64xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func @outlined(%a: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %z = arith.constant dense<0> : vector<16xi32>
    vector.transfer_write %z, %a[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}

// -----

// A herd of two tiles: each core calls the one clone, with its own buffers.
// CHECK-LABEL: aie.device
// CHECK: aie.core
// CHECK: call @tile_body(%{{.*}}, %{{.*}}) : (memref<64xi32>, memref<64xi32>) -> ()
// CHECK: aie.core
// CHECK: call @tile_body(%{{.*}}, %{{.*}}) : (memref<64xi32>, memref<64xi32>) -> ()
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32> {llvm.noalias}, %{{.*}}: memref<64xi32> {llvm.noalias})
// CHECK-NOT: func.func private @tile_body
// CHECK: }
module {
  func.func @two_tiles() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c2, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %in = memref.alloc() : memref<64xi32, 2>
          %out = memref.alloc() : memref<64xi32, 2>
          func.call @tile_body(%in, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}

// -----

// An LLVM intrinsic the cloned function calls is declared as it is.
// CHECK-LABEL: aie.device
// CHECK: call @llvm.aie2p.vshuffle
// CHECK: func.func private @llvm.aie2p.vshuffle(vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>{{$}}
module {
  func.func @intrinsic() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {link_with = "kernel.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<16xi32, 2>
          func.call @tile_body(%buf) : (memref<16xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<16xi32, 2>) {
    %c0 = arith.constant 0 : index
    %m = arith.constant 20 : i32
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<16xi32, 2>, vector<16xi32>
    %s = func.call @llvm.aie2p.vshuffle(%v, %v, %m) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
    vector.transfer_write %s, %a[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32, 2>
    return
  }
  func.func private @llvm.aie2p.vshuffle(vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
}
