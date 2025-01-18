import air
import air.compiler.util
from air.dialects import linalg, tensor, arith, func, memref
from air.ir import *
import air.passmanager
from air.dialects import air as airdialect
from air.compiler.util import run_transform
import sys

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Tiling
    ################################################

    air_tiled_ir_string = """
    #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
    #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
    #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
    module {
      func.func @forward(%arg0: memref<256x256xi32>, %arg1: memref<256x256xi32>) -> memref<256x256xi32> {
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c0_i32 = arith.constant 0 : i32
        %alloc = memref.alloc() : memref<256x256xi32>
        scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c256, %c256) step (%c64, %c64) {
          %subview = memref.subview %alloc[%arg2, %arg3] [64, 64] [1, 1] : memref<256x256xi32> to memref<64x64xi32, strided<[256, 1], offset: ?>>
          %alloc_0 = memref.alloc() : memref<1x1x64x64xi32, 1>
          scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c64, %c64) step (%c32, %c32) {
            %alloc_2 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
            linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<1x1x4x8x4x8xi32, 2>)
            %subview_3 = memref.subview %alloc_0[0, 0, %arg4, %arg5] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x64xi32, 1> to memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1>
            scf.for %arg6 = %c0 to %c256 step %c256 {
              %subview_5 = memref.subview %arg0[%arg2, %arg6] [64, 256] [1, 1] : memref<256x256xi32> to memref<64x256xi32, strided<[256, 1], offset: ?>>
              %subview_6 = memref.subview %arg1[%arg6, %arg3] [256, 64] [1, 1] : memref<256x256xi32> to memref<256x64xi32, strided<[256, 1], offset: ?>>
              %alloc_7 = memref.alloc() : memref<1x1x64x256xi32, 1>
              %alloc_8 = memref.alloc() : memref<1x1x256x64xi32, 1>
              air.dma_memcpy_nd (%alloc_7[] [] [], %subview_5[] [] []) : (memref<1x1x64x256xi32, 1>, memref<64x256xi32, strided<[256, 1], offset: ?>>)
              air.dma_memcpy_nd (%alloc_8[] [] [], %subview_6[] [] []) : (memref<1x1x256x64xi32, 1>, memref<256x64xi32, strided<[256, 1], offset: ?>>)
              scf.for %arg7 = %c0 to %c256 step %c32 {
                %subview_9 = memref.subview %alloc_7[0, 0, %arg4, %arg7] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x256xi32, 1> to memref<1x1x32x32xi32, strided<[16384, 16384, 256, 1], offset: ?>, 1>
                %subview_10 = memref.subview %alloc_8[0, 0, %arg7, %arg5] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x256x64xi32, 1> to memref<1x1x32x32xi32, strided<[16384, 16384, 64, 1], offset: ?>, 1>
                %alloc_11 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
                %alloc_12 = memref.alloc() : memref<1x1x4x4x8x8xi32, 2>
                %expand_shape = memref.expand_shape %subview_9 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 8, 4, 4, 8] : memref<1x1x32x32xi32, strided<[16384, 16384, 256, 1], offset: ?>, 1> into memref<1x1x8x4x4x8xi32, strided<[16384, 16384, 1024, 256, 8, 1], offset: ?>, 1>
                %transpose_13 = memref.transpose %expand_shape (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x4x4x8xi32, strided<[16384, 16384, 1024, 256, 8, 1], offset: ?>, 1> to memref<1x1x4x8x4x8xi32, strided<[16384, 16384, 8, 1024, 256, 1], offset: ?>, 1>
                air.dma_memcpy_nd (%alloc_11[] [] [], %transpose_13[] [] []) : (memref<1x1x4x8x4x8xi32, 2>, memref<1x1x4x8x4x8xi32, strided<[16384, 16384, 8, 1024, 256, 1], offset: ?>, 1>)
                %expand_shape_14 = memref.expand_shape %subview_10 [[0], [1], [2, 3], [4, 5]] output_shape [1, 1, 4, 8, 4, 8] : memref<1x1x32x32xi32, strided<[16384, 16384, 64, 1], offset: ?>, 1> into memref<1x1x4x8x4x8xi32, strided<[16384, 16384, 512, 64, 8, 1], offset: ?>, 1>
                %transpose_15 = memref.transpose %expand_shape_14 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x4x8x4x8xi32, strided<[16384, 16384, 512, 64, 8, 1], offset: ?>, 1> to memref<1x1x4x4x8x8xi32, strided<[16384, 16384, 8, 512, 64, 1], offset: ?>, 1>
                air.dma_memcpy_nd (%alloc_12[] [] [], %transpose_15[] [] []) : (memref<1x1x4x4x8x8xi32, 2>, memref<1x1x4x4x8x8xi32, strided<[16384, 16384, 8, 512, 64, 1], offset: ?>, 1>)
                linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_11, %alloc_12 : memref<1x1x4x8x4x8xi32, 2>, memref<1x1x4x4x8x8xi32, 2>) outs(%alloc_2 : memref<1x1x4x8x4x8xi32, 2>) {
                ^bb0(%in: i32, %in_16: i32, %out: i32):
                  %0 = arith.muli %in, %in_16 : i32
                  %1 = arith.addi %out, %0 : i32
                  linalg.yield %1 : i32
                }
                memref.dealloc %alloc_11 : memref<1x1x4x8x4x8xi32, 2>
                memref.dealloc %alloc_12 : memref<1x1x4x4x8x8xi32, 2>
              }
              memref.dealloc %alloc_7 : memref<1x1x64x256xi32, 1>
              memref.dealloc %alloc_8 : memref<1x1x256x64xi32, 1>
            }
            %transpose_4 = memref.transpose %alloc_2 (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x4x8x4x8xi32, 2> to memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 32, 8, 256, 1]>, 2>
            air.dma_memcpy_nd (%subview_3[] [] [], %transpose_4[] [] []) : (memref<1x1x32x32xi32, strided<[4096, 4096, 64, 1], offset: ?>, 1>, memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 32, 8, 256, 1]>, 2>)
            memref.dealloc %alloc_2 : memref<1x1x4x8x4x8xi32, 2>
            scf.reduce 
          }
          %subview_1 = memref.subview %alloc_0[0, 0, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x64x64xi32, 1> to memref<64x64xi32, 1>
          %transpose = memref.transpose %subview_1 (d0, d1) -> (d0, d1) : memref<64x64xi32, 1> to memref<64x64xi32, strided<[64, 1]>, 1>
          air.dma_memcpy_nd (%subview[] [] [], %transpose[] [] []) : (memref<64x64xi32, strided<[256, 1], offset: ?>>, memref<64x64xi32, strided<[64, 1]>, 1>)
          memref.dealloc %alloc_0 : memref<1x1x64x64xi32, 1>
          scf.reduce 
        }
        return %alloc : memref<256x256xi32>
      }
    }
    """
    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "buffer-results-to-out-params",
                "air-par-to-herd{depth=1}",
                "air-par-to-launch{has-air-segment=true}",
                "air-copy-to-dma",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

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
                "func.func(air-loop-fusion)",
                "air-label-scf-for-to-ping-pong",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    # Not sure why parsing the ir solves the segmentation fault...
    air_module = Module.parse(str(air_module))
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
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ################################################
    ## Place herd to segment
    ################################################

    air_async_module = Module.parse(str(air_module))
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
                "func.func(convert-linalg-to-loops)",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

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
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ################################################
    ## MLIR-AIR runtime lowering
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "func.func(air-opt-shim-dma-bds{device=npu2})",
                "air-to-std",
                "canonicalize",
                "func.func(affine-loop-opt{affine-opt-tile-sizes=4,4})",
                "func.func(air-unroll-outer-affine-loops{depth=2})",
                "affine-expand-index-ops",
                "airrt-to-npu",
                "canonicalize",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    with open("aie.mlir", "w") as f:
        f.write(str(air_module))
