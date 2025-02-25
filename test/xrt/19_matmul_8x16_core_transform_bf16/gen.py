# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from air.backend.xrt import XRTBackend
from air.ir import *
import air.passmanager


################################################
## Input SCF and Linalg IR
################################################

air_tiled_ir_string = """
#map = affine_map<(d0) -> (d0 * 128)>
#map2 = affine_map<(d0) -> (d0 * 64)>
#map3 = affine_map<()[s0] -> (s0 * 8)>
#map4 = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @forward(%arg0: memref<512x256xbf16>, %arg1: memref<32x8x8x4x16xbf16>, %arg2: memref<512x512xf32>) -> memref<512x512xf32> {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<2xindex>
    memref.store %c64, %alloc[%c0] : memref<2xindex>
    memref.store %c64, %alloc[%c1] : memref<2xindex>
    scf.forall (%arg3, %arg4) in (4, 4) {
      %0 = affine.apply #map(%arg3)
      %1 = affine.apply #map(%arg4)
      %subview = memref.subview %arg2[%0, %1] [128, 128] [1, 1] : memref<512x512xf32> to memref<128x128xf32, strided<[512, 1], offset: ?>>
      %alloc_0 = memref.alloc() : memref<128x256xbf16, 1>
      scf.for %arg5 = %c0 to %c256 step %c256 {
        %subview_3 = memref.subview %arg0[%0, %arg5] [128, 256] [1, 1] : memref<512x256xbf16> to memref<128x256xbf16, strided<[256, 1], offset: ?>>
        %subview_4 = memref.subview %alloc_0[0, %arg5] [128, 256] [1, 1] : memref<128x256xbf16, 1> to memref<128x256xbf16, strided<[256, 1], offset: ?>, 1>
        linalg.copy ins(%subview_3 : memref<128x256xbf16, strided<[256, 1], offset: ?>>) outs(%subview_4 : memref<128x256xbf16, strided<[256, 1], offset: ?>, 1>)
      }
      %alloc_1 = memref.alloc() : memref<32x8x8x16xbf16, 1>
      scf.for %arg5 = %c0 to %c32 step %c32 {
        %subview_3 = memref.subview %arg1[%arg5, 0, 0, %arg4, 0] [32, 8, 8, 1, 16] [1, 1, 1, 1, 1] : memref<32x8x8x4x16xbf16> to memref<32x8x8x16xbf16, strided<[4096, 512, 64, 1], offset: ?>>
        %subview_4 = memref.subview %alloc_1[%arg5, 0, 0, 0] [32, 8, 8, 16] [1, 1, 1, 1] : memref<32x8x8x16xbf16, 1> to memref<32x8x8x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>
        linalg.copy ins(%subview_3 : memref<32x8x8x16xbf16, strided<[4096, 512, 64, 1], offset: ?>>) outs(%subview_4 : memref<32x8x8x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>)
      }
      %alloc_2 = memref.alloc() : memref<128x128xf32, 1>
      scf.forall (%arg5, %arg6) in (2, 2) {
        %2 = affine.apply #map2(%arg5)
        %3 = affine.apply #map2(%arg6)
        %map = affine.apply #map4(%arg6)
        %subview_3 = memref.subview %alloc_2[%2, %3] [64, 64] [1, 1] : memref<128x128xf32, 1> to memref<64x64xf32, strided<[128, 1], offset: ?>, 1>
        %alloc_4 = memref.alloc() : memref<64x64xf32, 2>
        linalg.fill ins(%cst : f32) outs(%alloc_4 : memref<64x64xf32, 2>)
        scf.for %arg7 = %c0 to %c32 step %c8 {
          %4 = affine.apply #map3()[%arg7]
          %subview_5 = memref.subview %alloc_0[%2, %4] [64, 64] [1, 1] : memref<128x256xbf16, 1> to memref<64x64xbf16, strided<[256, 1], offset: ?>, 1>
          %subview_6 = memref.subview %alloc_1[%arg7, %map, 0, 0] [8, 4, 8, 16] [1, 1, 1, 1] : memref<32x8x8x16xbf16, 1> to memref<8x4x8x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>
          %transpose = memref.transpose %subview_6 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<8x4x8x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1> to memref<8x8x4x16xbf16, strided<[1024, 16, 128, 1], offset: ?>, 1>
          %alloc_7 = memref.alloc() : memref<64x64xbf16, 2>
          %alloc_8 = memref.alloc() : memref<8x8x4x16xbf16, 2>
          memref.copy %subview_5, %alloc_7 : memref<64x64xbf16, strided<[256, 1], offset: ?>, 1> to memref<64x64xbf16, 2>
          memref.copy %transpose, %alloc_8 : memref<8x8x4x16xbf16, strided<[1024, 16, 128, 1], offset: ?>, 1> to memref<8x8x4x16xbf16, 2>
          %reshape = memref.reshape %alloc_8(%alloc) : (memref<8x8x4x16xbf16, 2>, memref<2xindex>) -> memref<64x64xbf16, 2>
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_7, %reshape : memref<64x64xbf16, 2>, memref<64x64xbf16, 2>) outs(%alloc_4 : memref<64x64xf32, 2>)
          memref.dealloc %alloc_7 : memref<64x64xbf16, 2>
          memref.dealloc %alloc_8 : memref<8x8x4x16xbf16, 2>
        }
        memref.copy %alloc_4, %subview_3 : memref<64x64xf32, 2> to memref<64x64xf32, strided<[128, 1], offset: ?>, 1>
        memref.dealloc %alloc_4 : memref<64x64xf32, 2>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      linalg.copy ins(%alloc_2 : memref<128x128xf32, 1>) outs(%subview : memref<128x128xf32, strided<[512, 1], offset: ?>>)
      memref.dealloc %alloc_0 : memref<128x256xbf16, 1>
      memref.dealloc %alloc_1 : memref<32x8x8x16xbf16, 1>
      memref.dealloc %alloc_2 : memref<128x128xf32, 1>
    }
    return %arg2 : memref<512x512xf32>
  }
}
"""

context = Context()
air_module = Module.parse(air_tiled_ir_string, context)

################################################
## Binding parallel loops to air hierarchies
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-copy-to-dma",
            "air-par-to-herd{depth=-1}",
            "air-par-to-launch{has-air-segment=1}",
            "scf-forall-to-for",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)

pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

###############################################
# Run compile and load
###############################################

backend = XRTBackend(
    air_loop_fusion=True,
    lower_linalg_to_func="kernel.o",
)
module_function = backend.compile_and_load(air_module)
