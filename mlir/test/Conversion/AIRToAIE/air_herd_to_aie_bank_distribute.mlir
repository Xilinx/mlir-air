//===- air_herd_to_aie_bank_distribute.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that on AIE2/AIE2P targets the L1 buffers passed as different
// positional arguments of an extern (link_with) kernel call receive distinct
// `mem_bank` attributes, so the downstream `aie-assign-buffer-addresses`
// pass does not collapse them into a single bank.

// RUN: air-opt %s -air-to-aie='device=npu1 row-offset=2 col-offset=0' | FileCheck %s --check-prefix=NPU1
// RUN: air-opt %s -air-to-aie='device=xcvc1902' | FileCheck %s --check-prefix=AIE1

// NPU1-LABEL: aie.device(npu1)
// NPU1: %[[A:.*]] = aie.buffer({{.*}}) {{.*mem_bank = 0.*}} : memref<256xbf16, 2>
// NPU1: %[[B:.*]] = aie.buffer({{.*}}) {{.*mem_bank = 1.*}} : memref<256xbf16, 2>
// NPU1: %[[C:.*]] = aie.buffer({{.*}}) {{.*mem_bank = 2.*}} : memref<256xi32, 2>
// NPU1: aie.core
// NPU1:   call @matvec_kernel(%[[A]], %[[B]], %[[C]])

// AIE1: aie.device(xcvc1902)
// AIE1-NOT: mem_bank

module {

func.func private @matvec_kernel(memref<256xbf16, 2>, memref<256xbf16, 2>, memref<256xi32, 2>) attributes {link_with = "matvec.o", llvm.emit_c_interface}

func.func @test_bank_distribute() {
  %c1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %c1, %size_y = %c1) attributes {link_with = "matvec.o"} {
    %a = memref.alloc() : memref<256xbf16, 2>
    %b = memref.alloc() : memref<256xbf16, 2>
    %c = memref.alloc() : memref<256xi32, 2>
    func.call @matvec_kernel(%a, %b, %c) : (memref<256xbf16, 2>, memref<256xbf16, 2>, memref<256xi32, 2>) -> ()
  }
  return
}

}
