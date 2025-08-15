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
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def GetSegmentForOp : Op<Transform_Dialect, "air.get_segment_for",
    [NavigationTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Gets a handle to the parent 'air.segment' of the given operation";
  let description = [{
    Produces a handle to the parent `air.segment` op for each payload IR
    operation associated with the operand. Fails if a segment cannot be found.
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

def SegmentToAIEOp : Op<Transform_Dialect, "air.segment_to_aie",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Lower air.segment operations to mlir-aie modules.
  }];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::xilinx::air::SegmentOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// Ops implemented in mlir/lib/Conversion/ConvertToAIRPass.cpp
//

def ParToHerdOp : Op<Transform_Dialect, "air.par_to_herd",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Transform a `scf.parallel` operation into a `air.herd` operation.
    If the `scf.parallel` operation has more than two dimensions, then only
    the last two are used and a new `scf.parallel` is created outside of the
    herd. Returns the new `air.herd` operation.
  }];
  let arguments =
    (ins PDL_Operation:$target,
     DefaultValuedAttr<I64Attr, "0">:$first_dim);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::ParallelOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ParToLaunchOp : Op<Transform_Dialect, "air.par_to_launch",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Transform a `scf.parallel` operation into a `air.launch` operation.
    Returns the new `air.launch` operation.
  }];
  let arguments =
    (ins PDL_Operation:$target,
     DefaultValuedAttr<BoolAttr, "false">:$has_air_segment);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::ParallelOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ParToSegmentOp : Op<Transform_Dialect, "air.par_to_segment",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Transform a `scf.parallel` operation into a `air.segment` operation.
    Returns the new `air.segment` operation.
  }];
  let arguments =
    (ins PDL_Operation:$target,
     DefaultValuedAttr<BoolAttr, "false">:$has_air_segment);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::scf::ParallelOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def CopyToDmaOp : Op<Transform_Dialect, "air.copy_to_dma",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Transform a `memref.copy` operation into a `air.dma_memcpy_nd` operation.
    Returns the new `air.dma_memcpy_nd` operation.
  }];
  let arguments =
    (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::memref::CopyOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// Ops implemented in mlir/lib/Transform/AIRLinalgCodegen.cpp
//

def PipelineReduceOp : Op<Transform_Dialect, "air.pipeline_reduce",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    TransformOpInterface, TransformEachOpTrait]> {
  let description = [{
    Experimental
  }];
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
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def LinalgPromoteOp : Op<Transform_Dialect, "air.linalg_promote",
        [DeclareOpInterfaceMethods<TransformOpInterface>,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Promotes the specified operands of the target into a separate memory buffer
    using the `mlir::linalg::promoteSubViews` utility.

    This operation applies to Linalg ops that satisfy the
    `mlir::linalg::promoteSubviewsPrecondition`, otherwise it fails.

    When successful, several optimization passes are run on the resulting IR.
    The return handle points to the `target` operation that was modified
    inplace.

    The operation accepts as attributes the fields in
    `mlir::linalg::LinalgPromotionOptions`. In addition the memory space in
    allocated buffers can be specified with with the memory_space attribute as
    "L1", "L2" or "L3". The default memory space is L1.

    example:
    ```mlir
    %0 = transform.structured.match ops{["linalg.matmul"]} in %code  : (!pdl.operation) -> !pdl.operation
    %1 = transform.air.linalg_promote %0 {memory_space="L2", operands_to_promote=[0]}
    ```

    The `group_size` attribute is used to apply promotion to multiple
    linalg ops. When `group_size=N`, the `operands_to_promote` attribute refers to
    `N` payload operations at a time and the operand indices apply to the
    operands of the `N` operations in the order they appear in the target handle.

    For example,
    ```mlir
    %m = transform.structured.match ops{["linalg.matmul"]} in %f : (!pdl.operation) -> !pdl.operation
    %f = transform.structured.match ops{["linalg.fill"]} in %f : (!pdl.operation) -> !pdl.operation
    %h = transform.merge_handles %f, %m : !pdl.operation
    // promote the input of the fill operation and the output of the matmul operation to L1 memory
    transform.air.linalg_promote %h {"group_size"=2, "operands_to_promote"=[1,4], "memory_space"="L1"}
    ```
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$operands_to_promote,
                   DefaultValuedAttr<I64Attr, "1">:$group_size,
                   DefaultValuedAttr<BoolArrayAttr, "{}">:$use_full_tile_buffers,
                   UnitAttr:$use_full_tiles_by_default,
                   UnitAttr:$use_alloca,
                   OptionalAttr<I64Attr>:$alignment,
                   DefaultValuedAttr<StrAttr, "\"L1\"">:$memory_space);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

}

def LinalgTileOp : Op<Transform_Dialect, "air.linalg_tile",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Tile a linalg operation with the given sizes. The new linalg 
    operantion and the generated loop are returned. Tiling is 
    performed with the `transform::tileToForallOpImpl` so that an
    `scf.forall` loop is generated whenever possible.

    This is a variant of `transform.structured.tile_using_forall`.
  }];

  let arguments = (ins PDL_Operation:$target,
                   Variadic<PDL_Operation>:$dynamic_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sizes
                   );
  let results = (outs PDL_Operation:$tiled_linalg_op,
                      PDL_Operation:$loops);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<int64_t>":$staticTileSizes)>,
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedTileSizes)>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the list of tile sizes, which may be static (Attribute) or
    /// dynamic (Value).
    SmallVector<OpFoldResult> getMixedSizes();
  }];
}

def FuseIntoContainingMemrefOp :
    Op<Transform_Dialect, "air.fuse_into_containing_op",
      [DeclareOpInterfaceMethods<TransformOpInterface>,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

  let arguments = (ins PDL_Operation:$producer_op,
                       PDL_Operation:$containing_op);
  let results = (outs PDL_Operation:$fused_op);
  let assemblyFormat = "$producer_op `into` $containing_op attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$producerOp, "Value":$containingOp)>
  ];
}

// Ops implemented in mlir/lib/Transform/AIRLinalgBufferize.cpp
//

def AIRDuplicateAndEliminateEmptyTensorsOp
    : Op<Transform_Dialect, "air.duplicate_and_eliminate_empty_tensors",
        [DeclareOpInterfaceMethods<TransformOpInterface>,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    This transform op duplicates all multi-use `tensor.empty` operations
    within the target payload operation and then eliminates them via the
    One-Shot Bufferization "empty tensor elimination" step.

    The transform performs the following actions in sequence:

    1. **Duplicate**: For each `tensor.empty` with more than one use, clones
       the operation so that each use has a distinct producer.
    2. **Destination Style Conversion**: Applies Linalg conversion patterns
       to rewrite eligible operations into destination-passing style.
    3. **Bufferization Analysis**: Runs One-Shot Bufferization analysis with
       target-specific options to prepare for empty tensor elimination.
    4. **Elimination**: Replaces `tensor.empty` with buffer-backed allocations
       or forwards them to existing buffers, based on the analysis.

    This transform modifies the payload IR but does not change the transform
    IR except for reading the `target` handle.

    The `target` must be a handle to one or more operations containing
    `tensor.empty` producers that are eligible for bufferization.
  }];

  let arguments = (ins PDL_Operation:$target);

  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` type($target)";
}

def AIRBufferizeOp
    : Op<Transform_Dialect, "air.bufferize",
        [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
         DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Bufferizes the payload IR rooted at `target` using One-Shot Bufferization
    with AIR defaults, then performs a light cleanup.

    Specifically, this transform:
    - Runs One-Shot Bufferization on the target operation with AIR options
      (e.g., custom `allocationFn`/`memCpyFn`, disabled parallel-region checks).
      By default, allocations are created with `memref.alloc` and copies are
      emitted as `linalg.generic` elementwise copies to enable tiling/vectorization.
    - Applies canonical cleanups to erase unused operands and results that may
      appear after destination-passing rewrites.

    The transform **modifies the payload IR** but only **reads** the transform
    handle. On failure, emits a diagnostic attached to the target.

    **Operands**
    - `target` — A handle to the operation(s) to bufferize.

    **Results**
    - `result` — A handle to the same operation(s) after bufferization.

    This op is functional-style: the result type matches the handle type and
    the printer/parser use a functional signature:
      `air.bufferize %t : (!pdl.operation) -> !pdl.operation`
  }];

  let arguments = (
      ins PDL_Operation:$target
  );

  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "attr-dict $target `:` functional-type($target, results)";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];
}

#endif // AIR_TRANSFORM_OPS
