//===- AIRTransformOps.td ----------------------------------*- tablegen -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TRANSFORM_OPS
#define AIR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def GetPartitionForOp : Op<Transform_Dialect, "air.get_partition_for",
    [NavigationTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Gets a handle to the parent 'air.partition' of the given operation";
  let description = [{
    Produces a handle to the parent `air.partition` loop for each payload IR
    operation associated with the operand. Fails if a partition cannot be found.
    The list of operations associated with the handle contains
    parent operations in the same order as the list associated with the operand,
    except for operations that are parents to more than one input which are only
    present once.
  }];

  let arguments =
    (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$parent);

  let assemblyFormat = "$target attr-dict";
}

def PipelineReduceOp : Op<Transform_Dialect, "air.pipeline_reduce",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{}];
  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_size,
         DefaultValuedAttr<I64Attr, "1">:$pipeline_depth,
         DefaultValuedAttr<StrAttr, "\"horiz\"">:$direction,
         DefaultValuedAttr<UnitAttr, "false">:$promote);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict ";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def PartitionToAIEOp : Op<Transform_Dialect, "air.partition_to_aie",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Lower air.partition operations to mlir-aie modules.
  }];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);
  let assemblyFormat = "$target attr-dict";
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::xilinx::air::PartitionOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

#endif // AIR_TRANSFORM_OPS
