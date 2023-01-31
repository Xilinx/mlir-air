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
         UnitAttr:$promote);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict ";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ParToHerdOp : Op<Transform_Dialect, "air.par_to_herd",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{}];
  let arguments =
    (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ParallelOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ParToLaunchOp : Op<Transform_Dialect, "air.par_to_launch",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{}];
  let arguments =
    (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ParallelOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def CopyToDmaOp : Op<Transform_Dialect, "air.copy_to_dma",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{}];
  let arguments =
    (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::memref::CopyOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LinalgPromoteOp : Op<Transform_Dialect, "air.linalg_promote",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Promotes the specified operands of the target into a separate memory buffer.

    At this point, this transform does not allow customizing alloc/dealloc
    functions nor the behavior on copy in/out operations.

    #### Return modes

    This operation applies to a single Linalg op that satisfies the
    `promoteSubviewsPrecondition`, otherwise it fails.

    If the operations referred to by the `target` PDLOperation promote
    properly, the transform succeeds.

    When successful, the return handle points to the $target operation that
    was modified inplace.
  }];

  let arguments = (ins PDL_Operation:$target,
                       DefaultValuedAttr<I64ArrayAttr, "{}">:$operands_to_promote,
                       DefaultValuedAttr<BoolArrayAttr, "{}">:$use_full_tile_buffers,
                       UnitAttr:$use_full_tiles_by_default,
                       UnitAttr:$use_alloca,
                       OptionalAttr<I64Attr>:$alignment);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target,
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LinalgTileOp : Op<Transform_Dialect, "air.linalg_tile",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$dynamic_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$interchange
                   );
  let results = (outs PDL_Operation:$tiled_linalg_op,
                      Variadic<PDL_Operation>:$loops);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedTileSizes,
                   CArg<"ArrayRef<int64_t>", "{}">:$interchange)>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
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

def FuseIntoContainingMemrefOp :
    Op<Transform_Dialect, "air.fuse_into_containing_op",
      [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Fuse a producer into a containing operation.";

  let description = [{
    Fuses the `producer_op` into the `containing_op`.
    Returns a handle to the fused ops.

    The producer is a subview slice of a tiled op. This transform computes the
    accessed producer slice inside of the containing op ("tile and fuse").

    The containing op handle must be associated with exactly one payload op. The
    producer op handle may be associated with multiple payload ops. This
    transform fuses exactly one producer.

    #### Return modes

    If the producer could not be fused, this operation fails silently.
    This is the case when tiling fails or when the producer op has zero uses
    within the containing op. I.e., "producers" that are not consumed within the containing
    op are rejected by this operation.

    This operation reads and frees the producer handle.
    This operation reads the containing op handle.
  }];

  let arguments = (ins Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$producer_op,
                       Arg<PDL_Operation, "",
                           [TransformMappingRead]>:$containing_op);
  let results = (outs Res<PDL_Operation, "",
                          [TransformMappingAlloc,
                           TransformMappingWrite]>:$fused_op);
  let assemblyFormat = "$producer_op `into` $containing_op attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$producerOp, "Value":$containingOp)>
  ];
}
#endif // AIR_TRANSFORM_OPS
