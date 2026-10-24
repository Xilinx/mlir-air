//===- dma_to_channel_l3_memory_order.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -split-input-file | FileCheck %s

// L3 holds the tile's two column halves one after the other; L2 keeps it row
// by row. Copied in L3's order, the L3 side is one contiguous run and the
// reordering goes to the L2 side.

// CHECK-LABEL: func.func @read
// CHECK: air.channel.put {{.*}} (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c8192{{.*}}] [2, 32, 128] [4096, 128, 1]) : (memref<*xbf16>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[0, 0, 0] [2, 32, 128] [128, 256, 1]) : (memref<32x2x128xbf16, 1 : i32>)

module {
  func.func @read(%arg0: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0) : memref<*xbf16> {
      %1 = air.segment @seg async args(%s=%a) : memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c4096 = arith.constant 4096 : index
        %c8192 = arith.constant 8192 : index
        %t, %buf = air.execute -> (memref<32x2x128xbf16, 1 : i32>) {
          %b = memref.alloc() : memref<32x2x128xbf16, 1 : i32>
          air.execute_terminator %b : memref<32x2x128xbf16, 1 : i32>
        }
        %d = air.dma_memcpy_nd async [%t] (%buf[] [] [], %s[%c0, %c0, %c8192] [%c32, %c2, %c128] [%c128, %c4096, %c1_0]) : (memref<32x2x128xbf16, 1 : i32>, memref<*xbf16>)
      }
    }
    return
  }
}

// -----

// The same tile written back from L2.

// CHECK-LABEL: func.func @write
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c8192{{.*}}] [2, 32, 128] [4096, 128, 1]) : (memref<*xbf16>)
// CHECK: air.channel.put {{.*}} (%{{.*}}[0, 0, 0] [2, 32, 128] [128, 256, 1]) : (memref<32x2x128xbf16, 1 : i32>)

module {
  func.func @write(%arg0: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0) : memref<*xbf16> {
      %1 = air.segment @seg async args(%s=%a) : memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c4096 = arith.constant 4096 : index
        %c8192 = arith.constant 8192 : index
        %t, %buf = air.execute -> (memref<32x2x128xbf16, 1 : i32>) {
          %b = memref.alloc() : memref<32x2x128xbf16, 1 : i32>
          air.execute_terminator %b : memref<32x2x128xbf16, 1 : i32>
        }
        %d = air.dma_memcpy_nd async [%t] (%s[%c0, %c0, %c8192] [%c32, %c2, %c128] [%c128, %c4096, %c1_0], %buf[] [] []) : (memref<*xbf16>, memref<32x2x128xbf16, 1 : i32>)
      }
    }
    return
  }
}

// -----

// The halves are 8192 apart, past the 4096 of the first: in L3's order the
// copy is still not one run, so it stays as written.

// CHECK-LABEL: func.func @not_contiguous
// CHECK: air.channel.put {{.*}} [%c32{{.*}}, %c2{{.*}}, %c128{{.*}}] [%c128{{.*}}, %c8192{{.*}}, %c1{{.*}}]) : (memref<*xbf16>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[] [] []) : (memref<32x2x128xbf16, 1 : i32>)

module {
  func.func @not_contiguous(%arg0: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0) : memref<*xbf16> {
      %1 = air.segment @seg async args(%s=%a) : memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c8192 = arith.constant 8192 : index
        %t, %buf = air.execute -> (memref<32x2x128xbf16, 1 : i32>) {
          %b = memref.alloc() : memref<32x2x128xbf16, 1 : i32>
          air.execute_terminator %b : memref<32x2x128xbf16, 1 : i32>
        }
        %d = air.dma_memcpy_nd async [%t] (%buf[] [] [], %s[%c0, %c0, %c0] [%c32, %c2, %c128] [%c128, %c8192, %c1_0]) : (memref<32x2x128xbf16, 1 : i32>, memref<*xbf16>)
      }
    }
    return
  }
}

// -----

// In L3's order the L2 side would need a dim of 2048, more than a memtile
// descriptor counts to: the copy stays as written.

// CHECK-LABEL: func.func @too_long
// CHECK: air.channel.put {{.*}} [%c2048{{.*}}, %c2{{.*}}] [%c1{{.*}}, %c2048{{.*}}]) : (memref<*xbf16>)

module {
  func.func @too_long(%arg0: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0) : memref<*xbf16> {
      %1 = air.segment @seg async args(%s=%a) : memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c2048 = arith.constant 2048 : index
        %t, %buf = air.execute -> (memref<2048x2xbf16, 1 : i32>) {
          %b = memref.alloc() : memref<2048x2xbf16, 1 : i32>
          air.execute_terminator %b : memref<2048x2xbf16, 1 : i32>
        }
        %d = air.dma_memcpy_nd async [%t] (%buf[] [] [], %s[%c0, %c0] [%c2048, %c2] [%c1_0, %c2048]) : (memref<2048x2xbf16, 1 : i32>, memref<*xbf16>)
      }
    }
    return
  }
}

// -----

// The L2 side has its own pattern, here the second half of a larger buffer:
// its offsets and strides are permuted with the L3 side's.

// CHECK-LABEL: func.func @explicit_l2
// CHECK: air.channel.put {{.*}} (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [2, 32, 128] [4096, 128, 1]) : (memref<*xbf16>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c0{{.*}}, %c32{{.*}}, %c0{{.*}}] [2, 32, 128] [128, 256, 1]) : (memref<64x2x128xbf16, 1 : i32>)

module {
  func.func @explicit_l2(%arg0: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0) : memref<*xbf16> {
      %1 = air.segment @seg async args(%s=%a) : memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c4096 = arith.constant 4096 : index
        %t, %buf = air.execute -> (memref<64x2x128xbf16, 1 : i32>) {
          %b = memref.alloc() : memref<64x2x128xbf16, 1 : i32>
          air.execute_terminator %b : memref<64x2x128xbf16, 1 : i32>
        }
        %d = air.dma_memcpy_nd async [%t] (%buf[%c32, %c0, %c0] [%c32, %c2, %c128] [%c256, %c128, %c1_0], %s[%c0, %c0, %c0] [%c32, %c2, %c128] [%c128, %c4096, %c1_0]) : (memref<64x2x128xbf16, 1 : i32>, memref<*xbf16>)
      }
    }
    return
  }
}

// -----

// The L2 buffer is a strided view: its rows are not where its shape alone
// puts them, so the copy stays as written.

// CHECK-LABEL: func.func @strided_l2
// CHECK: air.channel.put {{.*}} [%c32{{.*}}, %c2{{.*}}, %c128{{.*}}] [%c128{{.*}}, %c4096{{.*}}, %c1{{.*}}]) : (memref<*xbf16>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[] [] [])

module {
  func.func @strided_l2(%arg0: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0) : memref<*xbf16> {
      %1 = air.segment @seg async args(%s=%a) : memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c4096 = arith.constant 4096 : index
        %t, %buf = air.execute -> (memref<32x2x128xbf16, strided<[512, 128, 1]>, 1 : i32>) {
          %b = memref.alloc() : memref<32x4x128xbf16, 1 : i32>
          %v = memref.subview %b[0, 0, 0] [32, 2, 128] [1, 1, 1] : memref<32x4x128xbf16, 1 : i32> to memref<32x2x128xbf16, strided<[512, 128, 1]>, 1 : i32>
          air.execute_terminator %v : memref<32x2x128xbf16, strided<[512, 128, 1]>, 1 : i32>
        }
        %d = air.dma_memcpy_nd async [%t] (%buf[] [] [], %s[%c0, %c0, %c0] [%c32, %c2, %c128] [%c128, %c4096, %c1_0]) : (memref<32x2x128xbf16, strided<[512, 128, 1]>, 1 : i32>, memref<*xbf16>)
      }
    }
    return
  }
}

// -----

// Five dims are more than a memtile descriptor takes: the copy stays as
// written.

// CHECK-LABEL: func.func @rank5
// CHECK: air.channel.put {{.*}} [%c8{{.*}}, %c16{{.*}}, %c4{{.*}}, %c2{{.*}}, %c1{{.*}}]) : (memref<*xbf16>)

module {
  func.func @rank5(%arg0: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0) : memref<*xbf16> {
      %1 = air.segment @seg async args(%s=%a) : memref<*xbf16> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %t, %buf = air.execute -> (memref<2x2x2x2x2xbf16, 1 : i32>) {
          %b = memref.alloc() : memref<2x2x2x2x2xbf16, 1 : i32>
          air.execute_terminator %b : memref<2x2x2x2x2xbf16, 1 : i32>
        }
        %d = air.dma_memcpy_nd async [%t] (%buf[] [] [], %s[%c0, %c0, %c0, %c0, %c0] [%c2, %c2, %c2, %c2, %c2] [%c8, %c16, %c4, %c2, %c1_0]) : (memref<2x2x2x2x2xbf16, 1 : i32>, memref<*xbf16>)
      }
    }
    return
  }
}

// -----

// A loop walks the L2 offset, one column block per trip. A later pass may
// shrink the buffer to one block, which drops the walked offset only from a
// row-major L2 side: the copy stays as written.

// CHECK-LABEL: func.func @loop_walked_l2
// CHECK: air.channel.put {{.*}} (%{{.*}}[%c0{{.*}}, %{{.*}}] [%c256{{.*}}, %c16{{.*}}] [%c1{{.*}}, %c256{{.*}}]) : (memref<*xf32>)
// CHECK: air.channel.get {{.*}} (%{{.*}}[%c0{{.*}}, %{{.*}}] [%c256{{.*}}, %c16{{.*}}] [%c32{{.*}}, %c1{{.*}}]) : (memref<256x32xf32, 1 : i32>)

module {
  func.func @loop_walked_l2(%arg0: memref<*xf32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%a=%arg0) : memref<*xf32> {
      %1 = air.segment @seg async args(%s=%a) : memref<*xf32> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %t, %buf = air.execute -> (memref<256x32xf32, 1 : i32>) {
          %b = memref.alloc() : memref<256x32xf32, 1 : i32>
          air.execute_terminator %b : memref<256x32xf32, 1 : i32>
        }
        %2 = scf.for %k = %c0 to %c32 step %c16 iter_args(%dep = %t) -> (!air.async.token) {
          %d = air.dma_memcpy_nd async [%dep] (%buf[%c0, %k] [%c256, %c16] [%c32, %c1_0], %s[%c0, %k] [%c256, %c16] [%c1_0, %c256]) : (memref<256x32xf32, 1 : i32>, memref<*xf32>)
          scf.yield %d : !air.async.token
        }
      }
    }
    return
  }
}
