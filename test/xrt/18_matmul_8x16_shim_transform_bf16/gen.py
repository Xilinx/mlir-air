# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air
import air.compiler.util
from air.dialects import linalg, arith, func
from air.ir import *
import air.passmanager
from air._mlir_libs._air import run_transform
from air.dialects.air import module_builder

import argparse


################################################
## Tiling
################################################

air_tiled_ir_string = """
#map = affine_map<(d0) -> (d0 * 128)>
#map2 = affine_map<(d0) -> (d0 * 64)>
#map3 = affine_map<()[s0] -> (s0 * 8)>
#map4 = affine_map<(d0) -> (d0 * 4)>
#map5 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @forward(%arg0: memref<512x256xbf16>, %arg1: memref<32x8x8x64xbf16>, %arg2: memref<512x512xf32>) -> memref<512x512xf32> {
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
      %map1 = affine.apply #map5(%arg4)
      %subview = memref.subview %arg2[%0, %1] [128, 128] [1, 1] : memref<512x512xf32> to memref<128x128xf32, strided<[512, 1], offset: ?>>
      %alloc_0 = memref.alloc() : memref<128x256xbf16, 1>
      scf.for %arg5 = %c0 to %c256 step %c256 {
        %subview_3 = memref.subview %arg0[%0, %arg5] [128, 256] [1, 1] : memref<512x256xbf16> to memref<128x256xbf16, strided<[256, 1], offset: ?>>
        %subview_4 = memref.subview %alloc_0[0, %arg5] [128, 256] [1, 1] : memref<128x256xbf16, 1> to memref<128x256xbf16, strided<[256, 1], offset: ?>, 1>
        linalg.copy ins(%subview_3 : memref<128x256xbf16, strided<[256, 1], offset: ?>>) outs(%subview_4 : memref<128x256xbf16, strided<[256, 1], offset: ?>, 1>)
      }
      %alloc_1 = memref.alloc() : memref<32x8x8x16xbf16, 1>
      scf.for %arg5 = %c0 to %c32 step %c32 {
        %subview_3 = memref.subview %arg1[%arg5, 0, 0, %map1] [32, 8, 8, 16] [1, 1, 1, 1] : memref<32x8x8x64xbf16> to memref<32x8x8x16xbf16, strided<[4096, 512, 64, 1], offset: ?>>
        %transpose = memref.transpose %subview_3 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<32x8x8x16xbf16, strided<[4096, 512, 64, 1], offset: ?>> to memref<32x8x8x16xbf16, strided<[4096, 64, 512, 1], offset: ?>>
        %subview_4 = memref.subview %alloc_1[%arg5, 0, 0, 0] [32, 8, 8, 16] [1, 1, 1, 1] : memref<32x8x8x16xbf16, 1> to memref<32x8x8x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>
        linalg.copy ins(%transpose : memref<32x8x8x16xbf16, strided<[4096, 64, 512, 1], offset: ?>>) outs(%subview_4 : memref<32x8x8x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>)
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
          %subview_6 = memref.subview %alloc_1[%arg7, 0, %map, 0] [8, 8, 4, 16] [1, 1, 1, 1] : memref<32x8x8x16xbf16, 1> to memref<8x8x4x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>
          %alloc_7 = memref.alloc() : memref<64x64xbf16, 2>
          %alloc_8 = memref.alloc() : memref<8x8x4x16xbf16, 2>
          memref.copy %subview_5, %alloc_7 : memref<64x64xbf16, strided<[256, 1], offset: ?>, 1> to memref<64x64xbf16, 2>
          memref.copy %subview_6, %alloc_8 : memref<8x8x4x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1> to memref<8x8x4x16xbf16, 2>
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

with open("air_tiled.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Binding parallel loops to air hierarchies
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-copy-to-dma",
            "air-linalg-to-func{link-with=kernel.o}",
            "air-par-to-herd{depth=1}",
            "air-par-to-launch{has-air-segment=1}",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)

pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

with open("air_sync.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Extract event dependency and optimize schedule
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-dependency",
            "air-dependency-schedule-opt",
            "air-specialize-dma-broadcast",
            "air-dma-to-channel",
            "canonicalize",
            "cse",
            "air-dependency-canonicalize",
            "canonicalize",
            "cse",
            "canonicalize",
            "cse",
            "func.func(air-loop-fusion)",
            "air-label-scf-for-to-ping-pong",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

with open("air_fusion.mlir", "w") as f:
    f.write(str(air_module))

# Not sure why parsing the ir solves the segmentation fault...
with open("air_fusion.mlir", "r") as f:
    air_module = f.read()

air_module = Module.parse(str(air_module), context=context)
pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-ping-pong-transform{keep-memref-dealloc=true}",
            "canonicalize",
            "cse",
            "air-specialize-channel-wrap-and-stride",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aircc_input.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Place herd to segment
################################################

air_async_module = Module.parse(str(air_module), context=context)
pipeline = (
    "builtin.module("
    + ",".join(
        [
            "func.func(air-collapse-herd)",
            "canonicalize",
            "cse",
            "air-place-herds{num-rows=4 num-cols=1 row-anchor=2 col-anchor=0}",
            "canonicalize",
            "cse",
            "func.func(air-renumber-dma)",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("air_placed.mlir", "w") as f:
    f.write(str(air_module))

################################################
## MLIR-AIR to MLIR-AIE
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-to-aie{row-offset=2 col-offset=1 device=npu2 emit-while-loop=true}",
            "canonicalize",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aircc_decomp_aiecc.mlir", "w") as f:
    f.write(str(air_module))

################################################
## MLIR-AIR runtime lowering
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "func.func(air-opt-shim-dma-bds{device=npu2})",
            "air-to-std",
            "symbol-dce",
            "canonicalize",
            "func.func(affine-loop-opt{affine-opt-tile-sizes=4,4})",
            "func.func(air-unroll-outer-affine-loops{depth=4})",
            "affine-expand-index-ops",
            "canonicalize",
            "cse",
            "airrt-to-npu",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aie.mlir", "w") as f:
    f.write(str(air_module))

import aie.compiler.aiecc.main as aiecc

aiecc_options = [
    "--no-aiesim",
    "--xchesscc",
    "--xbridge",
    "--aie-generate-cdo",
    "--aie-generate-npu",
    "--no-compile-host",
    "--npu-insts-name=insts.txt",
    "--xclbin-name=aie.xclbin",
    "aie.mlir",
]
aiecc.run(air_module, aiecc_options)
