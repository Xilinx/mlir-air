//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TRANSFORM_PASSES
#define AIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def AffineLoopOptPass : Pass<"affine-loop-opt", "func::FuncOp"> {
  let summary = "Affine loop transformations";
  let constructor = "xilinx::air::createAffineLoopOptPass()";
  let options = [
    ListOption<"clTileSizes", "affine-opt-tile-sizes", "unsigned",
               "Affine loop tiling sizes",
               "llvm::cl::ZeroOrMore">,
    ListOption<"clCopyDepths", "affine-opt-copy-depths", "unsigned",
               "Affine loop data copy loop depths",
               "llvm::cl::ZeroOrMore">,
    Option<"clFastSpace", "affine-opt-copy-fast-space", "unsigned", /*default=*/"1",
           "Fast memory space to use for affine data copy">,
    Option<"clSlowSpace", "affine-opt-copy-slow-space", "unsigned", 
            /*default=*/"0",
            "slow memory space to use for affine data copy">,
    Option<"clSeparate", "affine-opt-tile-separate", "bool", /*default=*/"false",
            "Affine loop tiling separates full and partial tiles">,
    Option<"clAffineOptLabel", "aaffine-opt-label", "std::string", /*default=*/"",
            "Transform loops with the given label">,
    Option<"clAffineOptPostLabel", "affine-opt-post-label", "std::string", /*default=*/"",
            "Label to apply to transformed loop nest">,
  ];
}

def AIRAutomaticTiling: Pass<"air-automatic-tiling", "func::FuncOp"> {
  let summary = "Tile loop nests manually or automatically with prime"
    " factorization";
  let constructor = "xilinx::air::createAIRAutomaticTilingPass()";
  let description = [{
    This pass performs multi-dimensional tiling of loop nests. If the tile 
    sizes are specified in the command line, all loops in loops nests will be 
    tiled along with the input factors. If no input tiling sizes are received, 
    the pass tiles all loops in loop bands with prime factors of the original 
    loop tripcounts. This pass assumes that loops are in normalized form and 
    the loop spaces are hyper-rectangular.

    Example 1: Manual tiling 

    `-air-automatic-tiling="loop-tile-sizes=64,2,2" -affine-simplify-structures -cse`

    Input:
    ```mlir
    module  {
      func.func @task(%arg0: tensor<4096xi32>, %arg1: tensor<4096xi32>) -> tensor<4096xi32> {
        %0 = memref.alloc() : memref<4096xi32>
        %1 = "aten.type_cast"(%arg0) : (tensor<4096xi32>) -> memref<4096xi32>
        %2 = "aten.type_cast"(%arg1) : (tensor<4096xi32>) -> memref<4096xi32>
        %c0 = constant 0 : index
        %c4096 = constant 4096 : index
        %c0_0 = constant 0 : index
        affine.for %arg2 = 0 to 4096 {
          %4 = affine.load %1[%arg2] : memref<4096xi32>
          %5 = affine.load %2[%arg2] : memref<4096xi32>
          %6 = muli %4, %5 : i32
          affine.store %6, %0[%arg2] : memref<4096xi32>
        } {affine_opt_label = "air.binary_op"}
        %3 = "aten.type_cast"(%0) : (memref<4096xi32>) -> tensor<4096xi32>
        return %3 : tensor<4096xi32>
      }
    }
    ```
    Output:
    ```mlir
    #map = affine_map<(d0, d1, d2, d3) -> (d0 + d1 * 64 + d2 * 128 + d3 * 256)>
    module  {
      func.func @task(%arg0: tensor<4096xi32>, %arg1: tensor<4096xi32>) -> tensor<4096xi32> {
        %0 = memref.alloc() : memref<4096xi32>
        %1 = "aten.type_cast"(%arg0) : (tensor<4096xi32>) -> memref<4096xi32>
        %2 = "aten.type_cast"(%arg1) : (tensor<4096xi32>) -> memref<4096xi32>
        affine.for %arg2 = 0 to 16 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 2 {
              affine.for %arg5 = 0 to 64 {
                %4 = affine.apply #map(%arg5, %arg4, %arg3, %arg2)
                %5 = affine.load %1[%4] : memref<4096xi32>
                %6 = affine.load %2[%4] : memref<4096xi32>
                %7 = muli %5, %6 : i32
                affine.store %7, %0[%4] : memref<4096xi32>
              }
            }
          }
        } {affine_opt_label = ""}
        %3 = "aten.type_cast"(%0) : (memref<4096xi32>) -> tensor<4096xi32>
        return %3 : tensor<4096xi32>
      }
    }
    ```

    Example 2: Automatic tiling

    `-air-automatic-tiling -affine-simplify-structures -cse`

    Input:

    ```mlir
    module  {
      func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
        %0 = memref.alloc() : memref<28x10xf32>
        %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
        %2 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
        affine.for %arg2 = 0 to 28 {
          affine.for %arg3 = 0 to 10 {
            %4 = affine.load %1[%arg2, %arg3] : memref<28x10xf32>
            %5 = affine.load %2[%arg2, %arg3] : memref<28x10xf32>
            %6 = mulf %4, %5 : f32
            affine.store %6, %0[%arg2, %arg3] : memref<28x10xf32>
          }
        } {affine_opt_label = "air.binary_op"}
        %3 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
        return %3 : tensor<28x10xf32>
      }
    }
    ```

    Output:

    ```mlir
    #map0 = affine_map<(d0, d1, d2) -> (d0 + d1 * 7 + d2 * 14)>
    #map1 = affine_map<(d0, d1) -> (d0 + d1 * 5)>
    module  {
      func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
        %0 = memref.alloc() : memref<28x10xf32>
        %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
        %2 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 7 {
              %4 = affine.apply #map0(%arg4, %arg3, %arg2)
              affine.for %arg5 = 0 to 2 {
                affine.for %arg6 = 0 to 5 {
                  %5 = affine.apply #map1(%arg6, %arg5)
                  %6 = affine.load %1[%4, %5] : memref<28x10xf32>
                  %7 = affine.load %2[%4, %5] : memref<28x10xf32>
                  %8 = mulf %6, %7 : f32
                  affine.store %8, %0[%4, %5] : memref<28x10xf32>
                }
              }
            }
          }
        }
        %3 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
        return %3 : tensor<28x10xf32>
      }
    }
    ```

  }];
  let options = [
    ListOption<"loopTileSizes", "loop-tile-sizes", "unsigned",
               "A list of multi-dimensional loop tile sizes",
               "llvm::cl::ZeroOrMore">,
    Option<"clTileSeparate", "tile-separate", "bool", /*default=*/"false",
           "AIR loop tiling separates full and partial tiles">,
    Option<"clLabel", "air-label", "std::string", /*default=*/"",
           "Transform loops with the given label">,
    Option<"clPostLabel", "air-post-label", "std::string", /*default=*/"",
           "Label to apply to transformed loop nest">
  ];
}

def AIRHerdAssign : Pass<"air-herd-assign", "ModuleOp"> {
  let summary = "Transfor affine.for to affine.parallel";
  let constructor = "xilinx::air::createAIRHerdAssignPass()";
}

def AIRHerdPlacementPass : Pass<"air-place-herds", "ModuleOp"> {
  let summary = "Places herds onto a segment.";
  let constructor = "xilinx::air::createAIRHerdPlacementPass()";

  let options = [
    Option<"clNumRows", "num-rows", "int", /*default=*/"8",
           "Number of rows of AIE tiles in a segment">,
    Option<"clNumCols", "num-cols", "int", /*default=*/"10",
           "Number of columns of AIE tiles in a segment">,
    Option<"clAnchorPointRow", "row-anchor", "int", /*default=*/"0",
           "Anchoring row number of segments">,
    Option<"clAnchorPointCol", "col-anchor", "int", /*default=*/"0",
           "Anchoring column number of segments">
  ];

  let description = [{
    This pass performs placement of air herds onto a segment with a 
    specific number of rows and columns. Assumes segment
    size (provided with an anchor point) will fit on physical board 
    dimensions. The placement starts at the bottom left of the segment 
    and tries to place the largest herd as it moves to the right side of 
    the row. If it can't place the largest herd remaining in a given tile, 
    it will try again with smaller and smaller herds. 

    Example with grid size set to 8 rows and 10 columns:

    `-air-place-herds"num-rows=8 num-cols=10 row-anchor=0 col-anchor=0"`

  Input: 
    ```mlir 
    #map0 = affine_map<()[s0] -> (s0 * 64)>
    #map1 = affine_map<()[s0] -> (s0 * 512)>
    #map2 = affine_map<()[s0] -> (s0 * 32)>
    module attributes {torch.debug_module_name = "mmult"} {
      func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
        %c16 = arith.constant 16 : index
        %c48 = arith.constant 48 : index
        %cst = arith.constant 0.000000e+00 : bf16
        %0 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
        linalg.fill ins(%cst : bf16) outs(%0 : memref<24576x1024xbf16>)
        %1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
        memref.copy %0, %1 : memref<24576x1024xbf16> to memref<24576x1024xbf16>
        %2 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
        air.launch @launch_0 (%arg2, %arg3) in (%arg4=%c48, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%1, %arg9=%2) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 6 : i64, size_y = 2 : i64} {
          air.segment @segment_0  args(%arg10=%arg2, %arg11=%arg3, %arg12=%arg4, %arg13=%arg5, %arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16>, memref<24576x1024xbf16> attributes {resource_type = "vckxyz", size_x = 3 : i64, size_y = 2 : i64} {
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c0 = arith.constant 0 : index
            %c1024 = arith.constant 1024 : index
            %c64 = arith.constant 64 : index
            %3 = affine.apply #map0()[%arg11]
            %4 = affine.apply #map1()[%arg10]
            scf.for %arg18 = %c0 to %c1024 step %c64 {
              %12 = memref.alloc() : memref<64x64xbf16, 1>
              %13 = memref.alloc() : memref<64x64xbf16, 1>
              %14 = memref.alloc() : memref<64x64xbf16, 1>
              air.dma_memcpy_nd (%12[] [] [], %arg14[%4, %arg18] [%c64, %c64] [%c1024, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
              air.dma_memcpy_nd (%13[] [] [], %arg15[%arg18, %3] [%c64, %c64] [%c1024, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<1024x1024xbf16>)
              air.dma_memcpy_nd (%14[] [] [], %arg16[%4, %3] [%c64, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<24576x1024xbf16>)
              air.herd @matmul_herd_0  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
                %c1_0 = arith.constant 1 : index
                %c0_1 = arith.constant 0 : index
                %c64_2 = arith.constant 64 : index
                %c32 = arith.constant 32 : index
                %15 = affine.apply #map2()[%arg19]
                %16 = affine.apply #map2()[%arg20]
                scf.for %arg26 = %c0_1 to %c64_2 step %c32 {
                  %17 = memref.alloc() : memref<32x32xbf16, 2>
                  %18 = memref.alloc() : memref<32x32xbf16, 2>
                  %19 = memref.alloc() : memref<32x32xbf16, 2>
                  air.dma_memcpy_nd (%17[] [] [], %arg23[%15, %arg26] [%c32, %c32] [%c64_2, %c1_0]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                  air.dma_memcpy_nd (%18[] [] [], %arg24[%arg26, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                  air.dma_memcpy_nd (%19[] [] [], %arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
                  linalg.matmul ins(%17, %18 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%19 : memref<32x32xbf16, 2>)
                  air.dma_memcpy_nd (%arg25[%15, %16] [%c32, %c32] [%c64_2, %c1_0], %19[] [] []) {id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
                  memref.dealloc %17 : memref<32x32xbf16, 2>
                  memref.dealloc %18 : memref<32x32xbf16, 2>
                  memref.dealloc %19 : memref<32x32xbf16, 2>
                }
                air.herd_terminator
              }
              air.dma_memcpy_nd (%arg16[%4, %3] [%c64, %c64] [%c1024, %c1], %14[] [] []) {id = 8 : i32} : (memref<24576x1024xbf16>, memref<64x64xbf16, 1>)
              memref.dealloc %12 : memref<64x64xbf16, 1>
              memref.dealloc %13 : memref<64x64xbf16, 1>
              memref.dealloc %14 : memref<64x64xbf16, 1>
            }
            air.segment_terminator
          }
          air.launch_terminator
        }
        return %2 : memref<24576x1024xbf16>
      }
    }
    ```
  output:
    ```mlir
    ....
    air.herd @matmul_herd_0  tile (%arg19, %arg20) in (%arg21=%c2, %arg22=%c2) args(%arg23=%12, %arg24=%13, %arg25=%14) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {x_loc = 0 : i64, y_loc = 7: i64} {
    ...
    ```
  }];
}

def AIRTransformInterpreterPass : Pass<"air-transform", "ModuleOp"> {
  let summary = "Transform IR with transform dialect";
  let constructor = "xilinx::air::createAIRTransformInterpreterPass()";
  let options = [
    Option<"clTransformFileName", "filename", "std::string",
            /*default=*/"", "Transform Dialect filename">
  ];
}

def AIRLinalgCodegen : Pass<"air-linalg-codegen", "ModuleOp"> {
  let summary = "AIR codegen strategies for linalg";
  let constructor = "xilinx::air::createAIRLinalgCodegenPass()";
  let description = [{
    This pass implements some tiling strategies for linalg ops targeting AIR
    dialect.
  }];
  let options = [
    ListOption<"clHerdSize", "herd-size", "unsigned",
               "Herd size to target", "llvm::cl::ZeroOrMore">,
    ListOption<"clL1TileSize", "l1-tile-size", "unsigned",
               "Tile factors to pass to L1 tiling",
               "llvm::cl::ZeroOrMore">,
    ListOption<"clL2TileSize", "l2-tile-size", "unsigned",
               "Tile factors to pass to L2 tiling",
               "llvm::cl::ZeroOrMore">,
    ListOption<"clL1TileInterchange", "l1-tile-permute", "unsigned",
               "Tile permute vector to pass to L1 tiling",
               "llvm::cl::ZeroOrMore">,
    ListOption<"clL2TileInterchange", "l2-tile-permute", "unsigned",
               "Tile permute vector to pass to L2 tiling",
               "llvm::cl::ZeroOrMore">,
    ListOption<"clL1OperandsToPromote", "l1-promote-operands", "unsigned",
               "Indices of subviews to promote",
               "llvm::cl::ZeroOrMore">,
    ListOption<"clL2OperandsToPromote", "l2-promote-operands", "unsigned",
               "Indices of subviews to promote",
               "llvm::cl::ZeroOrMore">,
    Option<"clL1Promote", "l1-promote", "bool", "true",
           "Promote tiles to L1 memory">,
    Option<"clL2Promote", "l2-promote", "bool", "true",
           "Promote tiles to L2 memory">,
    Option<"clL1MaxSize", "l1-size", "unsigned", "32768",
           "L1 allocation limit in bytes">,
    Option<"clL2MaxSize", "l2-size", "unsigned", "0",
           "L2 allocation limit in bytes">,
    Option<"clInputFilter", "input-filter", "std::string",
            /*default=*/"",
            "Input filter for linalg transformations">,
    Option<"clLinalgCodegenTestPatterns", "test-patterns", "bool",
            "false", "test patterns">

  ];
}

def AIRLinalgOpStats : Pass<"air-linalg-op-stats", "ModuleOp"> {
  let summary = "AIR linalg operation statistics";
  let constructor = "xilinx::air::createAIRLinalgOpStatsPass()";
  let description = [{
  }];
}

def AIRLowerLinalgTensors : Pass<"air-lower-linalg-tensors", "ModuleOp"> {
  let summary = "Lowering from linalg on tensors to loops";
  let constructor = "xilinx::air::createAIRLowerLinalgTensorsPass()";
  let description = [{
    This pass implements a lowering pipeline from linalg on tensors to affine
    loops. There are three stages:
    1. Bufferize with `linalg::populateLinalgBufferizePatterns`
    2. Run some peephole optimizations to cleanup after bufferization.
    3. Named ops are transformed to linalg GenericOps then all GenericOps
    are lowered to loops using `linalg::LinalgLoweringPattern`.

    The transforms are biased toward aie.core regions and are intended
    to be run after the air-to-aie pass.
  }];
}

def AIRRegularizeLoop: Pass<"air-regularize-loop", "func::FuncOp"> {
  let summary = "Move operations inside the innermost loop body to regularize "
    "loop nests";
  let constructor = "xilinx::air::createAIRRegularizeLoopPass()";
  let description = [{
    This pass regularizes loop nests by moving intermediate operations between
    subloops in a loop nest inside the innermost loop body. The pass is 
    essentiallythe inverse of the affine loop invariant code motion pass. For 
    each opeation that makes the loop nest non-perfect, the pass will check 
    recursively if the content of the operation is independent of the induction 
    variable of the inner loop. And if it is independent, the operation will be 
    moved inside the inner loop body until the induction variable of the inner 
    loop is dependent on the operation or there are no loops at the same level.

    Example: Regularize a loop nest with `-air-regularize-loop`

    Input:

    ```mlir
    #map0 = affine_map<(d0, d1, d2) -> (d0 + d1 * 7 + d2 * 14)>
    #map1 = affine_map<(d0, d1) -> (d0 + d1 * 5)>
    module  {
      func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
        %0 = memref.alloc() : memref<28x10xf32>
        %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
        %2 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 7 {
              %4 = affine.apply #map0(%arg4, %arg3, %arg2)
              affine.for %arg5 = 0 to 2 {
                affine.for %arg6 = 0 to 5 {
                  %5 = affine.apply #map1(%arg6, %arg5)
                  %6 = affine.load %1[%4, %5] : memref<28x10xf32>
                  %7 = affine.load %2[%4, %5] : memref<28x10xf32>
                  %8 = mulf %6, %7 : f32
                  affine.store %8, %0[%4, %5] : memref<28x10xf32>
                }
              }
            }
          }
        }
        %3 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
        return %3 : tensor<28x10xf32>
      }
    }
    ```

    Output:

    ```mlir
    #map0 = affine_map<(d0, d1, d2) -> (d0 + d1 * 7 + d2 * 14)>
    #map1 = affine_map<(d0, d1) -> (d0 + d1 * 5)>
    module  {
      func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
        %0 = memref.alloc() : memref<28x10xf32>
        %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
        %2 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 7 {
              affine.for %arg5 = 0 to 2 {
                affine.for %arg6 = 0 to 5 {
                  %4 = affine.apply #map0(%arg4, %arg3, %arg2)
                  %5 = affine.apply #map1(%arg6, %arg5)
                  %6 = affine.load %1[%4, %5] : memref<28x10xf32>
                  %7 = affine.load %2[%4, %5] : memref<28x10xf32>
                  %8 = mulf %6, %7 : f32
                  affine.store %8, %0[%4, %5] : memref<28x10xf32>
                }
              }
            }
          }
        }
        %3 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
        return %3 : tensor<28x10xf32>
      }
    }
    ```

  }];
}

def AIRReturnElimination : Pass<"air-return-elimination", "ModuleOp"> {
  let summary = "Convert functions to return their values with out parameters";
  let constructor = "xilinx::air::createReturnEliminationPass()";
}

def AIRLoopPermutation: Pass<"air-loop-permutation", "func::FuncOp"> {
  let summary = "Change the loop ordering according to the input mapping";
  let constructor = "xilinx::air::createAIRLoopPermutationPass()";
  let description = [{
    This pass performs a loop nest reordering according to the input mapping. 
    The i-th loop will be moved from position i -> permMap[i] where the counting 
    of i starts at the outermost loop. The pass transforms only
    perfect loop nests. The specified ordering starts from 0, and should be of 
    the same length as the loop nest size. Each number is required to appear once in 
    the input mapping.

    Example: Permute a loop nest with `-air-loop-permutation="loop-order=4,3,2,1,0"`

    Input: 

    ```mlir
    #map0 = affine_map<(d0, d1, d2) -> (d0 + d1 * 7 + d2 * 14)>
    #map1 = affine_map<(d0, d1) -> (d0 + d1 * 5)>
    module  {
      func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
        %0 = memref.alloc() : memref<28x10xf32>
        %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
        %2 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 7 {
              affine.for %arg5 = 0 to 2 {
                affine.for %arg6 = 0 to 5 {
                  %4 = affine.apply #map0(%arg4, %arg3, %arg2)
                  %5 = affine.apply #map1(%arg6, %arg5)
                  %6 = affine.load %1[%4, %5] : memref<28x10xf32>
                  %7 = affine.load %2[%4, %5] : memref<28x10xf32>
                  %8 = mulf %6, %7 : f32
                  affine.store %8, %0[%4, %5] : memref<28x10xf32>
                }
              }
            }
          }
        }
        %3 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
        return %3 : tensor<28x10xf32>
      }
    }
    ```

    Output:

    ```mlir
    #map0 = affine_map<(d0, d1, d2) -> (d0 + d1 * 7 + d2 * 14)>
    #map1 = affine_map<(d0, d1) -> (d0 + d1 * 5)>
    module  {
      func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
        %0 = memref.alloc() : memref<28x10xf32>
        %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
        %2 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
        affine.for %arg2 = 0 to 5 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 7 {
              affine.for %arg5 = 0 to 2 {
                affine.for %arg6 = 0 to 2 {
                  %4 = affine.apply #map0(%arg4, %arg5, %arg6)
                  %5 = affine.apply #map1(%arg2, %arg3)
                  %6 = affine.load %1[%4, %5] : memref<28x10xf32>
                  %7 = affine.load %2[%4, %5] : memref<28x10xf32>
                  %8 = mulf %6, %7 : f32
                  affine.store %8, %0[%4, %5] : memref<28x10xf32>
                }
              }
            }
          }
        }
        %3 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
        return %3 : tensor<28x10xf32>
      }
    }
    ```

  }];
  let options = [
    ListOption<"loopOrder", "loop-order", "unsigned",
               "The target loop permutation ordering",
               "llvm::cl::OneOrMore">,
    Option<"clLabel", "air-label", "std::string", /*default=*/"",
           "Transform loops with the given label">,
    Option<"clPostLabel", "air-post-label", "std::string", 
            /*default=*/"",
            "Label to apply to transformed loop nest">
  ];
}

def AIRLoopMergingPass: Pass<"air-loop-merging", "func::FuncOp"> {
  let summary = "Merge several nested subloops into a single loop";
  let constructor = "xilinx::air::createAIRLoopMergingPass()";
  let description = [{
    This pass transforms several perfectly nested subloops into a single 
    loop. The trip count of the new single loop is the product of all
    trip counts in subloops. The original loop induction variables are 
    restored using floordiv and modulo operations. Users can specify which 
    loop levels they want to merge together.

    Example: Merge subloops with `-air-loop-merging="loop-merge-levels=1,2,3" -affine-simplify-structures -cse`

    Input:

    ```mlir
    #map0 = affine_map<(d0, d1, d2) -> (d0 + d1 * 7 + d2 * 14)>
    #map1 = affine_map<(d0, d1) -> (d0 + d1 * 5)>
    module  {
      func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
        %0 = memref.alloc() : memref<28x10xf32>
        %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
        %2 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 7 {
              affine.for %arg5 = 0 to 2 {
                affine.for %arg6 = 0 to 5 {
                  %4 = affine.apply #map0(%arg4, %arg3, %arg2)
                  %5 = affine.apply #map1(%arg6, %arg5)
                  %6 = affine.load %1[%4, %5] : memref<28x10xf32>
                  %7 = affine.load %2[%4, %5] : memref<28x10xf32>
                  %8 = mulf %6, %7 : f32
                  affine.store %8, %0[%4, %5] : memref<28x10xf32>
                }
              }
            }
          }
        } {affine_opt_label = ""}
        %3 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
        return %3 : tensor<28x10xf32>
      }
    }
    ```

    Output:

    ```mlir
    #map0 = affine_map<(d0, d1) -> (d0 * 14 + d1 floordiv 2 - ((d1 floordiv 2) floordiv 7) * 7 + (d1 floordiv 14) * 7)>
    #map1 = affine_map<(d0, d1) -> (d0 + d1 * 5 - (d1 floordiv 2) * 10)>
    module  {
      func.func @task(%arg0: tensor<28x10xf32>, %arg1: tensor<28x10xf32>) -> tensor<28x10xf32> {
        %0 = memref.alloc() : memref<28x10xf32>
        %1 = "aten.type_cast"(%arg0) : (tensor<28x10xf32>) -> memref<28x10xf32>
        %2 = "aten.type_cast"(%arg1) : (tensor<28x10xf32>) -> memref<28x10xf32>
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 28 {
            affine.for %arg4 = 0 to 5 {
              %4 = affine.apply #map0(%arg2, %arg3)
              %5 = affine.apply #map1(%arg4, %arg3)
              %6 = affine.load %1[%4, %5] : memref<28x10xf32>
              %7 = affine.load %2[%4, %5] : memref<28x10xf32>
              %8 = mulf %6, %7 : f32
              affine.store %8, %0[%4, %5] : memref<28x10xf32>
            }
          }
        } {affine_opt_label = ""}
        %3 = "aten.type_cast"(%0) : (memref<28x10xf32>) -> tensor<28x10xf32>
        return %3 : tensor<28x10xf32>
      }
    }
    ```

  }];
}

def AIRDependency : Pass<"air-dependency", "ModuleOp"> {
  let summary = "AIR dependency analysis";
  let constructor = "xilinx::air::createAIRDependencyPass()";
  let description = [{
    This pass analyzes dependencies among air.async_region ops, and constructs
    the dependency relationship between asynchronous events for scheduling. 
    The pass also generates a dot file which visualizes the dependency graph.

    Example 1: Simple data depdendency tracing

    `-air-dependency`

    Input:

    ```mlir
    module  {
      func.func @foo(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
        %c1 = arith.constant 1 : index
        %0 = memref.alloc() : memref<1024xi32, 1>
        air.launch_herd tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%0, %arg7=%arg1) : memref<1024xi32, 1>,memref<1024xi32> {
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %1 = memref.alloc() : memref<16xi32, 2>
          air.dma_memcpy (%1, %arg6, [%c0], [%c16], %c16) {id = 1 : i32} : (memref<16xi32, 2>, memref<1024xi32, 1>, [index], [index], index) -> ()
          air.dma_memcpy (%arg6, %1, [%c16], [%c0], %c16) {id = 2 : i32} : (memref<1024xi32, 1>, memref<16xi32, 2>, [index], [index], index) -> ()
          air.herd_terminator
        }
        memref.dealloc %0 : memref<1024xi32, 1>
        return
      }
    }
    ```

    Output:

    ```mlir
    module {
      func.func @foo(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
        %c1 = arith.constant 1 : index
        %asyncToken, %valOut = air.execute async  {
          %1 = memref.alloc() : memref<1024xi32, 1>
          air.execute_terminator %1 : memref<1024xi32, 1>
        } {id = 1 : i32} : (memref<1024xi32, 1>)
        %0 = air.launch_herd async  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%valOut, %arg7=%arg1) : memref<1024xi32, 1>, memref<1024xi32> attributes {id = 1 : i32} {
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %asyncToken_1, %valOut_2 = air.execute async  {
            %3 = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %3 : memref<16xi32, 2>
          } {id = 2 : i32} : (memref<16xi32, 2>)
          %1 = air.dma_memcpy async [%asyncToken_1] (%valOut_2, %arg6, [%c0], [%c16], %c16) {id = 1 : i32} : (memref<16xi32, 2>, memref<1024xi32, 1>, [index], [index], index) -> ()
          %2 = air.dma_memcpy async [%1] (%arg6, %valOut_2, [%c16], [%c0], %c16) {id = 2 : i32} : (memref<1024xi32, 1>, memref<16xi32, 2>, [index], [index], index) -> ()
          air.herd_terminator
        }
        %asyncToken_0 = air.execute async [%0, %asyncToken]  : (!air.async.token, !air.async.token) {
          memref.dealloc %valOut : memref<1024xi32, 1>
          air.execute_terminator
        } {id = 3 : i32}
        return
      }
    }
    ```

    Example 2: Loop-carried depdendency tracing

    `-air-dependency`

    Input:

    ```mlir
    #map = affine_map<()[s0] -> (s0 * 32)>
    module attributes {torch.debug_module_name = "mmult"} {
      func.func @forward(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
        %c1024 = arith.constant 1024 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %0 = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xf32>
        %1 = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xf32>
        linalg.fill ins(%cst : f32) outs(%0 : memref<1024x1024xf32>)
        memref.copy %0, %1 : memref<1024x1024xf32> to memref<1024x1024xf32>
        scf.for %arg3 = %c0 to %c1024 step %c128 {
          scf.for %arg4 = %c0 to %c1024 step %c128 {
            scf.for %arg5 = %c0 to %c1024 step %c32 {
              air.launch_herd  tile (%arg6, %arg7) in (%arg8=%c4, %arg9=%c4) args(%arg10=%arg3, %arg11=%arg5, %arg12=%arg0, %arg13=%arg4, %arg14=%arg1, %arg15=%1) : index, index, memref<1024x1024xf32>, index, memref<1024x1024xf32>, memref<1024x1024xf32> attributes {sym_name = "herd_0"} {
                %c1 = arith.constant 1 : index
                %c1024_0 = arith.constant 1024 : index
                %c32_1 = arith.constant 32 : index
                %2 = affine.apply #map()[%arg6]
                %3 = affine.apply #map()[%arg7]
                %4 = arith.addi %arg10, %2 : index
                %5 = arith.addi %arg13, %3 : index
                %6 = memref.alloc() : memref<32x32xf32, 2>
                %7 = memref.alloc() : memref<32x32xf32, 2>
                %8 = memref.alloc() : memref<32x32xf32, 2>
                air.dma_memcpy_nd (%6[] [] [], %arg12[%4, %arg11] [%c32_1, %c32_1] [%c1024_0, %c1]) {id = 1 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
                air.dma_memcpy_nd (%7[] [] [], %arg14[%arg11, %5] [%c32_1, %c32_1] [%c1024_0, %c1]) {id = 2 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
                air.dma_memcpy_nd (%8[] [] [], %arg15[%4, %5] [%c32_1, %c32_1] [%c1024_0, %c1]) {id = 3 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
                linalg.matmul ins(%6, %7 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%8 : memref<32x32xf32, 2>)
                air.dma_memcpy_nd (%arg15[%4, %5] [%c32_1, %c32_1] [%c1024_0, %c1], %8[] [] []) {id = 4 : i32} : (memref<1024x1024xf32>, memref<32x32xf32, 2>)
                memref.dealloc %6 : memref<32x32xf32, 2>
                memref.dealloc %7 : memref<32x32xf32, 2>
                memref.dealloc %8 : memref<32x32xf32, 2>
                air.herd_terminator
              }
            }
            scf.yield
          }
        }
        memref.copy %1, %arg2 : memref<1024x1024xf32> to memref<1024x1024xf32>
        return
      }
    }
    ```

    Output:

    ```mlir
    #map = affine_map<()[s0] -> (s0 * 32)>
    module attributes {torch.debug_module_name = "mmult"} {
      func.func @forward(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
        %c1024 = arith.constant 1024 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %asyncToken, %valOut = air.execute async  {
          %2 = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xf32>
          air.execute_terminator %2 : memref<1024x1024xf32>
        } {id = 1 : i32} : (memref<1024x1024xf32>)
        %asyncToken_0, %valOut_1 = air.execute async  {
          %2 = memref.alloc() {alignment = 128 : i64} : memref<1024x1024xf32>
          air.execute_terminator %2 : memref<1024x1024xf32>
        } {id = 2 : i32} : (memref<1024x1024xf32>)
        %asyncToken_2 = air.execute async [%asyncToken]  : (!air.async.token) {
          linalg.fill ins(%cst : f32) outs(%valOut : memref<1024x1024xf32>)
          air.execute_terminator
        } {id = 3 : i32}
        %asyncToken_3 = air.execute async [%asyncToken_0, %asyncToken_2]  : (!air.async.token, !air.async.token) {
          memref.copy %valOut, %valOut_1 : memref<1024x1024xf32> to memref<1024x1024xf32>
          air.execute_terminator
        } {id = 4 : i32}
        %0 = air.wait_all async [%asyncToken_3]  {id = 6 : i32}
        %1 = scf.for %arg3 = %c0 to %c1024 step %c128 iter_args(%arg4 = %0) -> (!air.async.token) {
          %c0_5 = arith.constant 0 : index
          %c1024_6 = arith.constant 1024 : index
          %c128_7 = arith.constant 128 : index
          %2 = air.wait_all async [%arg4]  {id = 4 : i32}
          %3 = scf.for %arg5 = %c0_5 to %c1024_6 step %c128_7 iter_args(%arg6 = %2) -> (!air.async.token) {
            %c0_8 = arith.constant 0 : index
            %c1024_9 = arith.constant 1024 : index
            %c32_10 = arith.constant 32 : index
            %5 = air.wait_all async [%arg6]  {id = 2 : i32}
            %6 = scf.for %arg7 = %c0_8 to %c1024_9 step %c32_10 iter_args(%arg8 = %5) -> (!air.async.token) {
              %c4_11 = arith.constant 4 : index
              %8 = air.launch_herd async [%arg8]  tile (%arg9, %arg10) in (%arg11=%c4_11, %arg12=%c4_11) args(%arg13=%arg3, %arg14=%arg7, %arg15=%arg0, %arg16=%arg5, %arg17=%arg1, %arg18=%valOut_1) : index, index, memref<1024x1024xf32>, index, memref<1024x1024xf32>, memref<1024x1024xf32> attributes {id = 1 : i32, sym_name = "herd_0"} {
                %c1 = arith.constant 1 : index
                %c1024_12 = arith.constant 1024 : index
                %c32_13 = arith.constant 32 : index
                %asyncToken_14, %valOut_15 = air.execute async  {
                  %14 = affine.apply #map()[%arg9]
                  air.execute_terminator %14 : index
                } {id = 5 : i32} : (index)
                %asyncToken_16, %valOut_17 = air.execute async  {
                  %14 = affine.apply #map()[%arg10]
                  air.execute_terminator %14 : index
                } {id = 6 : i32} : (index)
                %asyncToken_18, %valOut_19 = air.execute async [%asyncToken_14]  : (!air.async.token) {
                  %14 = arith.addi %arg13, %valOut_15 : index
                  air.execute_terminator %14 : index
                } {id = 7 : i32} : (index)
                %asyncToken_20, %valOut_21 = air.execute async [%asyncToken_16]  : (!air.async.token) {
                  %14 = arith.addi %arg16, %valOut_17 : index
                  air.execute_terminator %14 : index
                } {id = 8 : i32} : (index)
                %asyncToken_22, %valOut_23 = air.execute async  {
                  %14 = memref.alloc() : memref<32x32xf32, 2>
                  air.execute_terminator %14 : memref<32x32xf32, 2>
                } {id = 9 : i32} : (memref<32x32xf32, 2>)
                %asyncToken_24, %valOut_25 = air.execute async  {
                  %14 = memref.alloc() : memref<32x32xf32, 2>
                  air.execute_terminator %14 : memref<32x32xf32, 2>
                } {id = 10 : i32} : (memref<32x32xf32, 2>)
                %asyncToken_26, %valOut_27 = air.execute async  {
                  %14 = memref.alloc() : memref<32x32xf32, 2>
                  air.execute_terminator %14 : memref<32x32xf32, 2>
                } {id = 11 : i32} : (memref<32x32xf32, 2>)
                %10 = air.dma_memcpy_nd async [%asyncToken_22, %asyncToken_18] (%valOut_23[] [] [], %arg15[%valOut_19, %arg14] [%c32_13, %c32_13] [%c1024_12, %c1]) {id = 1 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
                %11 = air.dma_memcpy_nd async [%asyncToken_24, %asyncToken_20] (%valOut_25[] [] [], %arg17[%arg14, %valOut_21] [%c32_13, %c32_13] [%c1024_12, %c1]) {id = 2 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
                %12 = air.dma_memcpy_nd async [%asyncToken_26, %asyncToken_20, %asyncToken_18] (%valOut_27[] [] [], %arg18[%valOut_19, %valOut_21] [%c32_13, %c32_13] [%c1024_12, %c1]) {id = 3 : i32} : (memref<32x32xf32, 2>, memref<1024x1024xf32>)
                %asyncToken_28 = air.execute async [%11, %12, %10]  : (!air.async.token, !air.async.token, !air.async.token) {
                  linalg.matmul ins(%valOut_23, %valOut_25 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%valOut_27 : memref<32x32xf32, 2>)
                  air.execute_terminator
                } {id = 12 : i32}
                %13 = air.dma_memcpy_nd async [%asyncToken_28] (%arg18[%valOut_19, %valOut_21] [%c32_13, %c32_13] [%c1024_12, %c1], %valOut_27[] [] []) {id = 4 : i32} : (memref<1024x1024xf32>, memref<32x32xf32, 2>)
                %asyncToken_29 = air.execute async [%asyncToken_28]  : (!air.async.token) {
                  memref.dealloc %valOut_23 : memref<32x32xf32, 2>
                  air.execute_terminator
                } {id = 13 : i32}
                %asyncToken_30 = air.execute async [%asyncToken_28]  : (!air.async.token) {
                  memref.dealloc %valOut_25 : memref<32x32xf32, 2>
                  air.execute_terminator
                } {id = 14 : i32}
                %asyncToken_31 = air.execute async [%13]  : (!air.async.token) {
                  memref.dealloc %valOut_27 : memref<32x32xf32, 2>
                  air.execute_terminator
                } {id = 15 : i32}
                air.herd_terminator
              }
              %9 = air.wait_all async [%8]  {id = 1 : i32}
              scf.yield %9 : !air.async.token
            }
            %7 = air.wait_all async [%6]  {id = 3 : i32}
            scf.yield %7 : !air.async.token
          }
          %4 = air.wait_all async [%3]  {id = 5 : i32}
          scf.yield %4 : !air.async.token
        }
        %asyncToken_4 = air.execute async [%1]  : (!air.async.token) {
          memref.copy %valOut_1, %arg2 : memref<1024x1024xf32> to memref<1024x1024xf32>
          air.execute_terminator
        } {id = 16 : i32}
        return
      }
    }
    ```

  }];
}

def AIRHoistDmaInAccumPattern: Pass<"air-hoist-dma-in-accum-pattern", "ModuleOp"> {
  let summary = "Hoist pairs of DMA ops out of for loop based on dependency graph";
  let constructor = "xilinx::air::createAIRHoistDmaInAccumPattern()";
  let description = [{
    This pass detects redundant DMA operations in scf.for loops based on AIR event
    dependency generated by -air-dependency pass, and optimize the for loop's
    performance by hoisting them out of for loop's body.
  }];
}

def AIRBroadcastDetection: Pass<"air-broadcast-detection", "ModuleOp"> {
  let summary = "Detect DMA broadcast opportunities";
  let constructor = "xilinx::air::createAIRBroadcastDetection()";
  let description = [{
    This pass detects DMA broadcast opportunities by tracing the source indices'
    dependence to the induction variables of any parent spatial loop space. Upon
    successful detection, the DMA shall be annotated by an affine set attribute 
    named 'broadcast_pattern'. 
  }];
}

def AIRPruneLinalgGenericInputDma: Pass<"air-prune-linalg-generic-input-dma", "ModuleOp"> {
  let summary = "Detect and prune redundant DMA into linalg generic";
  let constructor = "xilinx::air::createAIRPruneLinalgGenericInputDma()";
  let description = [{
    This pass detects and prunes redundant DMA which copies into linalg generic
    input operands. 
  }];
}

def AIRAnnotateFrontAndBackOpsInForPattern: Pass<"air-annotate-front-and-back-ops-in-for-pattern", "ModuleOp"> {
  let summary = "Annotates ops in for loop body which are at the front and back of the body's dependency graph";
  let constructor = "xilinx::air::createAIRAnnotateFrontAndBackOpsInForPattern()";
  let description = [{
    This pass analyzes the loop body of an asynchronous for loop and annotates the 
    asynchronous operations which are located at the front and back of the loop
    body's dependency tree. This pass is used in the ping-pong pattern transformation 
    to detect the insertion and exit points in data producer and consumer sub-trees.
  }];
}

def AIRHoistMemallocInForPattern: Pass<"air-hoist-alloc-in-for-pattern", "ModuleOp"> {
  let summary = "Hoist pairs of alloc and dealloc ops out of for loop";
  let constructor = "xilinx::air::createAIRHoistMemallocInForPattern()";
  let description = [{
    This pass hoists pairs of `alloc` and `dealloc` ops out of a for loop body, to 
    represent the static allocation of memories. This pass is used in the ping-pong
    pattern transformation to identify and isolate the statically allocated ping and 
    pong buffers.
  }];
  let options = [
    Option<"clKeepMemrefDealloc", "keep-memref-dealloc", "bool", /*default=*/"false",
            "Flag to keep memref dealloc ops after transformation. Memref dealloc is used in air-to-aie pass as handle to generate lock releases.">
  ];
}

def AIRUnrollLoopForPipeliningPattern: Pass<"air-unroll-loop-for-pipelining-pattern", "ModuleOp"> {
  let summary = "Unroll loop by an integer factor";
  let constructor = "xilinx::air::createAIRUnrollLoopForPipeliningPattern()";
  let description = [{
    This pass unrolls a loop by an integer factor. This pass is used in the ping-pong
    pattern transformation to unroll a scf.for loop by 2 to ensure explicit
    representation of ping and pong processes, respectively.
  }];
}

def AIRConstructPingPongDependencyPattern: Pass<"air-construct-ping-pong-dependency-pattern", "ModuleOp"> {
  let summary = "Transform an scf.for loop into ping-pong pattern";
  let constructor = "xilinx::air::createAIRConstructPingPongDependencyPattern()";
  let description = [{
    This pass transform an `scf.for` loop into ping-pong pattern, by constructing 
    dependency edges to connect explicitly unrolled and annotated data producer and 
    consumer processes for ping and pong buffers, respectively. The dependency edges,
    being yielded across loop iterations, directly represent a compute scheduling 
    scheme which leads to concurrency between communication and compute in the form of 
    ping-pong buffering.
  }];
}

def AIRHoistOpsNotUsingPingPongPattern: Pass<"air-hoist-ops-not-using-ping-pong", "ModuleOp"> {
  let summary = "Hoists ops which are not direct users of the target memref";
  let constructor = "xilinx::air::createAIRHoistOpsNotUsingPingPongPattern()";
  let description = [{
    This pass isolates an `scf.for` loop in preparation for ping-pong transformation,
    by identifying child operations which are not direct consumers or producers of the 
    memref targetted for ping-pong buffering, and hoisting said operations out of the
    scf.for loop.
  }];
}

def AIRPingPongTransformationPattern: Pass<"air-ping-pong-transform", "ModuleOp"> {
  let summary = "Lower to pipelining pattern";
  let constructor = "xilinx::air::createAIRPingPongTransformationPattern()";
  let description = [{
    This pass lowers to pipelining pattern. This pass looks for the target ping and 
    pong buffers, and a surrounding scf.for loop, to construct explicit dependency 
    edges which represent a ping-pong buffering scheduling.
  }];
  let options = [
    Option<"clKeepMemrefDealloc", "keep-memref-dealloc", "bool", /*default=*/"false",
            "Flag to keep memref dealloc ops after transformation. Memref dealloc is used in air-to-aie pass as handle to generate lock releases.">
  ];
}

def AIRLabelScfForLoopForPingPongPattern: Pass<"air-label-scf-for-to-ping-pong", "ModuleOp"> {
  let summary = "Label all candidate scf.for loops for ping-pong transformation";
  let constructor = "xilinx::air::createAIRLabelScfForLoopForPingPongPattern()";
  let description = [{
    This pass labels all scf.for loops which contain air.execute event of memref.alloc,
    which is a direct child op of said scf.for, as candidate loop for ping-pong
    transformation. The label includes an attribute added to the child memref.alloc ops
    for subsequent hoisting, and an attribute added to the scf.for with an unroll factor.
  }];
}

def AIRLabelScfForLoopInAIRSegmentPattern: Pass<"air-label-scf-for-in-segment", "ModuleOp"> {
  let summary = "Label all candidate scf.for loops within air.segment for unrolling";
  let constructor = "xilinx::air::createAIRLabelScfForLoopInAIRSegmentPattern()";
  let description = [{
    This pass labels all scf.for loops contained in air.segment ops, except for those
    which are also contained in air.herd ops, with an 'unroll' integer attribute. Scf.for
    loops labelled with this attribute shall be unrolled, with the provided factor, by
    any subsequent 'air-unroll-loop-for-pipelining-pattern' pass.
  }];
}

def AIRSpecializeChannelWrapAndStridePattern: Pass<"air-specialize-channel-wrap-and-stride", "ModuleOp"> {
  let summary = "Specialize air.channel op in perfect loop nest with wraps and strides";
  let constructor = "xilinx::air::createAIRSpecializeChannelWrapAndStridePattern()";
  let description = [{
    This pass specialize air.channel op in perfect loop nest with wraps and strides, for
    efficient mapping to hardware buffer descriptors.
  }];
}

def AIRUnrollChannelByFactorPattern: Pass<"air-unroll-channel-by-factor", "ModuleOp"> {
  let summary = "Unroll channel puts and gets by an integer factor";
  let constructor = "xilinx::air::createAIRUnrollChannelByFactorPattern()";
  let description = [{
    This pass unrolls all puts and gets to an air.channel by an integer factor, so as to
    represent the usage of multiple physical DMA channels in parallel for improved
    available bandwidth for this data movement.
  }];
  let options = [
    Option<"clChanName", "channel-name", "std::string", 
            /*default=*/"\"\"",
            "Target channel to unroll.">,
    Option<"clUnrollDim", "unroll-dim", "int", /*default=*/"0",
           "Dimension id to unroll.">,
    Option<"clUnrollFactor", "unroll-factor", "int", /*default=*/"1",
           "Integer unroll factor.">
  ];
}

def AIREnforceLoopCarriedMemrefDeallocPattern: Pass<"air-enforce-loop-carried-memref-dealloc", "ModuleOp"> {
  let summary = "Enforce memref dealloc ops in loop iterations";
  let constructor = "xilinx::air::createAIREnforceLoopCarriedMemrefDeallocPattern()";
  let description = [{
    This pass enforces memref deallocation events to happen within each loop
    iteration, by connecting them in the loop-carried dependency path.
  }];
}

def AIRDeAliasMemref: Pass<"air-dealias-memref", "ModuleOp"> {
  let summary = "De-alias a memref into multiple memrefs";
  let constructor = "xilinx::air::createAIRDeAliasMemref()";
  let description = [{
    This pass detects memrefs which can de-alias into multiple memrefs over time, and
    generates extra copies of this memref. This process can improve the stability of the
    memref-to-buffer mappings.
  }];
}

def AIRFuseChannels: Pass<"air-fuse-channels", "ModuleOp"> {
  let summary = "Fuse multiple air.channel ops into one";
  let constructor = "xilinx::air::createAIRFuseChannels()";
  let description = [{
    This pass fuses multiple air.channel ops into one. The condition for fusing channels 
    is such that the puts and gets of all candidate channels mush share the same control 
    loop hierarchy, where every parent loops must have matching loop bounds.
  }];
}

def AIRIsolateAsyncDmaLoopNests: Pass<"air-isolate-async-dma-loop-nests", "ModuleOp"> {
  let summary = "Hoist dma ops into perfectly nested loop";
  let constructor = "xilinx::air::createAIRIsolateAsyncDmaLoopNests()";
  let description = [{
    This pass isolates loops containing dma memcpy ops into perfectly nested loops, by 
    hoisting them out of their previous parent loop.
  }];
}

def AIRDependencyScheduleOpt: Pass<"air-dependency-schedule-opt", "ModuleOp"> {
  let summary = "Optimize scheduling based on air async dependency";
  let constructor = "xilinx::air::createAIRDependencyScheduleOptPass()";
  let description = [{
    This pass contains multiple passes which optimize the schedule based on the
    dependency graph generated from -air-dependency pass.
  }];
}

def AIRDependencyCanonicalize: Pass<"air-dependency-canonicalize", "ModuleOp"> {
  let summary = "Canonicalize the dependency graph";
  let constructor = "xilinx::air::createAIRDependencyCanonicalizePass()";
  let description = [{
    This pass optimizes the dependency graph in air by removing non-dominant
    dependency edges.
  }];
  let options = [
    Option<"clDumpGraph", "dump-graph", "bool", /*default=*/"false",
            "Dump post-canonicalization dot graphs.">,
    Option<"clDumpDir", "output-dir", "std::string", 
            /*default=*/"\"\"",
            "Target directory to dump dot graphs.">
  ];
}

def AIRDependencyParseGraph: Pass<"air-dependency-parse-graph", "ModuleOp"> {
  let summary = "Parse the dependency graph and dump dot files";
  let constructor = "xilinx::air::createAIRDependencyParseGraphPass()";
  let description = [{
    This pass parses the dependency graph into an internal format, and dump
    dot files for graph visualization.
  }];
  let options = [
    Option<"clDumpDir", "output-dir", "std::string", 
            /*default=*/"\"\"",
            "Target directory to dump dot graphs.">,
    Option<"clShowCores", "show-cores", "bool", /*default=*/"false",
            "Show the graph of each AIE core.">
  ];
}

def AIRExamplePass : Pass<"air-example-pass", "ModuleOp"> {
  let summary = "Skeleton module op pass";
  let constructor = "xilinx::air::createAIRExamplePass()";
}

def AIRSpecializeDma : Pass<"air-specialize-dma", "ModuleOp"> {
  let summary = "Specialize dma operations";
  let constructor = "xilinx::air::createAIRSpecializeDma()";
}

def AIRSpecializeDmaBroadcast : Pass<"air-specialize-dma-broadcast", "ModuleOp"> {
  let summary = "Specialize dma operations for broadcast pattern";
  let constructor = "xilinx::air::createAIRSpecializeDmaBroadcast()";
}

def AIRPromoteUniformL1Dma : Pass<"air-promote-dma", "ModuleOp"> {
  let summary = "promote uniform dma operations";
  let constructor = "xilinx::air::createAIRPromoteUniformL1Dma()";
}

def AIRLinalgNamePass : Pass<"air-linalg-name", "ModuleOp"> {
  let summary = "Give linalg ops a LinalgTransformMarker string attribute if they don't already have one";
  let constructor = "xilinx::air::createAIRLinalgNamePass()";
}

def AIRRemoveLinalgNamePass : Pass<"air-rm-linalg-name", "ModuleOp"> {
  let summary = "Remove LinalgTransformMarker string attributes from linalg ops";
  let constructor = "xilinx::air::createAIRRemoveLinalgNamePass()";
}

def AIRPipelineReducePass : Pass<"air-pipeline-reduce", "func::FuncOp"> {
  let summary = "Turn a reduction dimension into a herd pipeline";
  let constructor = "xilinx::air::createAIRPipelineReducePass()";
  let options = [
    ListOption<"clTileSize", "tile-size", "unsigned",
               "Tile factors to pass to L1 tiling",
               "llvm::cl::ZeroOrMore">,
    Option<"clPipelineDepth", "pipeline-depth", "int", /*default=*/"4",
           "Pipeline depth to generate">,
    Option<"clPipelineDirection", "pipeline-direction", "std::string", 
            /*default=*/"\"horiz\"",
            "Pipeline direction attribute to use. Can be 'vert' or 'horiz'">,
    Option<"clPromoteSubViews", "promote", "bool", /*default=*/"false",
            "Promote subviews to memory buffers and insert copies.">
  ];
}

def AIRFuseParallelHerdPass : Pass<"air-fuse-parallel-launch", "ModuleOp"> {
  let summary = "fuse parallel launch pass";
  let constructor = "xilinx::air::createAIRFuseParallelHerdPass()";
}


def AIRRenumberDmaIdPass : Pass<"air-renumber-dma", "func::FuncOp"> {
  let summary = "Renumber air dma op ids";
  let constructor = "xilinx::air::createAIRRenumberDmaIdPass()";
  let options = [
    Option<"clMode", "mode", "std::string", /*default=*/"\"global\"",
           "In which hierarchy level to renumber the dma ops">,
  ];
}

def AIRLowerHerdParallelPass : Pass<"air-lower-herd-parallel", "func::FuncOp"> {
  let summary = "Remove scf.parallel from inside herds by transforming them to "
                "scf.for.";
  let constructor = "xilinx::air::createAIRLowerHerdParallelPass()";
}

def AIRLabelBroadcastChannelWithTilePass : Pass<"air-label-broadcast-channel-with-tile", "func::FuncOp"> {
  let summary = "Label broadcasted channel ops with tile coordinates.";
  let constructor = "xilinx::air::createAIRLabelBroadcastChannelWithTilePass()";
}

def AIRCollapseHerdPass : Pass<"air-collapse-herd", "func::FuncOp"> {
  let summary = "Collapse a multi-dimensional air.herd into a single column.";
  let constructor = "xilinx::air::createAIRCollapseHerdPass()";
}

def AIRUnrollOuterPerfectlyNestedLoopsPass : Pass<"air-unroll-outer-affine-loops", "func::FuncOp"> {
  let summary = "Unroll loops in a perfectly nested affine for loop nest, outer to inner.";
  let constructor = "xilinx::air::createAIRUnrollOuterPerfectlyNestedLoopsPass()";
  let options = [
    Option<"clDepth", "depth", "int", /*default=*/"1",
           "The number of outermost loops in the loop nest to unroll">,
  ];
}

#endif // AIR_CONVERSION_PASSES
