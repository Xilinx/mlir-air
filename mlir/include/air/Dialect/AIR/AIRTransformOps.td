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

def LinalgToLibraryCallOp : Op<Transform_Dialect, "air.linalg_to_library_call",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformOpInterface, TransformEachOpTrait]> {
  let summary = "Convert a linalg op to a function call (library call)";
  let description = [{
    Replaces a linalg op with a call to a function. If the `function_name`
    attribute is provided, it is used as the function name. Otherwise, the
    linalg op's `library_call` attribute is used. The function is created if
    it does not exist. If the `link_with` attribute is provided, it is used
    to link the function call to a prebuilt object that contains the 
    implementation of the function. If the linalg op is inside a herd, the 
    `link_with` attribute is propagated to the herd.

    Example:
    ```
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %f : (!pdl.operation) -> !pdl.operation
    %call = transform.air.linalg_to_library_call %matmul { function_name = "my_matmul", link_with = "extern_func.o" } : (!pdl.operation) -> !pdl.operation
    ```
  }];
  let arguments = (ins PDL_Operation:$target,
                   OptionalAttr<StrAttr>:$function_name,
                   OptionalAttr<StrAttr>:$link_with);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
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

def ForallWithReduceToParallelOp : Op<Transform_Dialect, "air.forall_with_reduce_to_parallel",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Converts a pattern of scf.forall and linalg.reduce to scf.parallel";
  let description = [{
    .
  }];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs Variadic<PDL_Operation>:$transformed);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
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

def RemoveUninitializedCopyOp : Op<Transform_Dialect, "air.remove_uninitialized_copy",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Remove copy operations that copy from uninitialized memrefs";
  let description = [{
    This transform walks through a func.func operation and identifies memref.copy 
    and linalg.copy operations where the source is an uninitialized memref 
    (allocated but not written to). Such copy operations are erased as they copy 
    undefined data.
    
    The transform detects the pattern where:
    1. A memref is allocated with memref.alloc
    2. A subview of that memref is created (optional)
    3. The memref/subview is used as source in memref.copy or linalg.copy before any write operations
    
    Returns a handle to the modified function.
    
    Examples:
    ```mlir
    // memref.copy case
    %alloc = memref.alloc() : memref<2x16x8xi32, 1>
    %subview = memref.subview %alloc[0, 0, 0] [1, 16, 8] [1, 1, 1] : ...
    %target = memref.alloc() : memref<1x16x8xi32, 2>
    memref.copy %subview, %target  // <- This copy will be erased
    
    // linalg.copy case
    %alloc2 = memref.alloc() : memref<16x8xi32, 1>
    %target2 = memref.alloc() : memref<16x8xi32, 2>
    linalg.copy ins(%alloc2 : memref<16x8xi32, 1>) outs(%target2 : memref<16x8xi32, 2>)  // <- This copy will be erased
    ```
  }];
  
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";
}

def EliminateCascadeMemcpyOp : Op<Transform_Dialect, "air.eliminate_cascade_memcpy",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Eliminate intermediate memref buffers in cascaded DMA operations";
  let description = [{
    This transform identifies and eliminates intermediate memref buffers in cascaded 
    air.dma_memcpy_nd operations. It looks for the pattern where an intermediate buffer
    is used exactly twice: once as the destination of a DMA operation and once as the
    source of another DMA operation, with both operations using default access patterns
    (empty offsets, sizes, and strides).
    
    The transformation replaces:
    ```mlir
    air.dma_memcpy_nd (%intermediate[] [] [], %source[] [] []) : (memref<...>, memref<...>)
    air.dma_memcpy_nd (%dest[] [] [], %intermediate[] [] []) : (memref<...>, memref<...>)
    ```
    
    With:
    ```mlir
    air.dma_memcpy_nd (%dest[] [] [], %source[] [] []) : (memref<...>, memref<...>)
    ```
    
    This optimization eliminates unnecessary intermediate memory allocations and reduces
    memory traffic, which is particularly beneficial for cascade patterns in AIR programs.
    
    Returns a handle to the modified operation.
  }];
  
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";
}

def ConvertMemrefCopyToLinalgCopyOp : Op<Transform_Dialect, "air.convert_memref_copy_to_linalg_copy",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Convert memref.copy operations to linalg.copy operations";
  let description = [{
    This transform converts `memref.copy` operations to `linalg.copy` operations.
    This can be useful for enabling further linalg-based optimizations and transformations.
    
    The transformation replaces:
    ```mlir
    memref.copy %source, %dest : memref<...> to memref<...>
    ```
    
    With:
    ```mlir
    linalg.copy ins(%source : memref<...>) outs(%dest : memref<...>)
    ```
    
    Returns a handle to the modified operation containing the transformed copies.
  }];
  
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";
}

// Ops implemented in mlir/lib/Transform/AIRLinalgBufferize.cpp
//

def AIRHoistStaticAllocOp :  Op<Transform_Dialect, "air.hoist_static_alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let summary = [{Hoist static allocations.}];
  let description = [{
    Moves certain statically-sized `memref.alloc` operations from inner blocks
    to the entry block of the target function. This shortens and unifies buffer
    lifetimes, which can unlock reuse and downstream optimizations.

    ### Notes / limitations

    * Currently targets `memref.alloc` buffers with static shapes.
    * Uses that require exact type equality across region boundaries (e.g.,
      `scf.yield`, `func.return`) are **not** rewritten; such allocations are skipped.
    * Hoisting increases the buffer's lifetime; apply with care on large buffers.

    ### Example

    Before:
    ```mlir
    func.func @foo(%arg0: memref<64xi32>) {
      scf.for %i = %c0 to %c4 step %c1 {
        %tmp = memref.alloc() : memref<64xi32>
        linalg.fill ins(%cst : i32) outs(%tmp : memref<64xi32>)
        memref.dealloc %tmp : memref<64xi32>
      }
      return
    }
    ```

    After:
    ```mlir
    func.func @foo(%arg0: memref<64xi32>) {
      %tmp.hoisted = memref.alloc() : memref<64xi32>
      scf.for %i = %c0 to %c4 step %c1 {
        linalg.fill ins(%cst : i32) outs(%tmp.hoisted : memref<64xi32>)
      }
      memref.dealloc %tmp.hoisted : memref<64xi32>
      return
    }
    ```

    ### Usage (Transform dialect)

    ```mlir
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb0(%f: !pdl.operation):
      transform.air.hoist_static_alloc %f
        : (!pdl.operation) -> ()
    }
    ```
  }];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::FunctionOpInterface funcOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // AIR_TRANSFORM_OPS
