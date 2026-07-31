//===- AIRRtOps.td -----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_OPS
#define AIRRT_OPS

include "air/Dialect/AIRRt/AIRRtBase.td"
include "mlir/IR/SymbolInterfaces.td"

class AIRRt_Op<string mnemonic, list<Trait> traits = []>
    : Op<AIRRt_Dialect, mnemonic, traits> {
}

def AIRRt_ModuleMetadataOp : AIRRt_Op<"module_metadata", [
  SingleBlockImplicitTerminator<"ModuleMetadataTerminatorOp">
]> {
  let summary = "Global metadata for the module";
  let description = [{
    This op contains a region containing airrt.segment_metadata ops,
    which give information about the segments in the module.

    There is one of these per module.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$segments);
  let hasCustomAssemblyFormat = 1;
}

def AIRRt_ModuleMetadataTerminatorOp
    : AIRRt_Op<"module_metadata_terminator",
               [Terminator, HasParent<"ModuleMetadataOp">]> {
  let summary = "Implicit terminator for ModuleMetadataOp's region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def AIRRt_SegmentMetadataOp : AIRRt_Op<"segment_metadata", [
  HasParent<"ModuleMetadataOp">,
  SingleBlockImplicitTerminator<"SegmentMetadataTerminatorOp">
]> {
  let summary = "Runtime metadata for one air region";
  let description = [{
    This op contains a region containing airrt.herd_metadata ops,
    which give information about the herds in the module.
  }];
  let arguments = (ins
    StrAttr:$sym_name
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$herds);
  let hasCustomAssemblyFormat = 1;
}

def AIRRt_SegmentMetadataTerminatorOp
    : AIRRt_Op<"segment_metadata_terminator",
               [Terminator, HasParent<"SegmentMetadataOp">]> {
  let summary = "Implicit terminator for SegmentMetadataOp's region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def AIRRt_HerdMetadataOp
    : AIRRt_Op<"herd_metadata", [
        HasParent<"SegmentMetadataOp">,
        SingleBlockImplicitTerminator<"HerdMetadataTerminatorOp">
      ]> {
  let summary = "Runtime metadata for a single herd";
  let description = [{
    Runtime metadata for a single herd.

  }];
  let arguments = (ins
    StrAttr:$sym_name
  );
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def AIRRt_HerdMetadataTerminatorOp
    : AIRRt_Op<"herd_metadata_terminator",
               [Terminator, HasParent<"HerdMetadataOp">]> {
  let summary = "Implicit terminator for HerdMetadataOp's region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def AIRRt_HerdLoadOp : AIRRt_Op<"herd_load", []> {
    let summary = "load a herd";
    let arguments = (ins StrAttr:$sym_name, Variadic<AnyType>:$rtp);
    let results = (outs I64:$h, Optional<AIRRt_Event>:$event);
    let assemblyFormat = [{
      $sym_name ` ``(` $rtp `)` attr-dict `:` functional-type($rtp, results)
    }];
    let hasCanonicalizer = 1;
}

def AIRRt_SegmentLoadOp : AIRRt_Op<"segment_load", []> {
    let summary = "load a segment";
    let arguments = (ins StrAttr:$sym_name);
    let results = (outs I64:$p, Optional<AIRRt_Event>:$event);
    let assemblyFormat = [{
      $sym_name attr-dict `:` type($p) (`,` type($event)^)?
    }];
    let hasCanonicalizer = 1;
}

def AIRRt_DmaMemcpyNdOp: AIRRt_Op<"dma_memcpy_nd", [AttrSizedOperandSegments]> {
  let summary = "dma operator";
  let arguments = (
    ins I32:$id,
        I64:$x,
        I64:$y,
        AnyRankedOrUnrankedMemRef:$memref,
        Variadic<I64>:$offsets,
        Variadic<I64>:$lengths,
        Variadic<I64>:$strides,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_offsets,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_lengths,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_strides
  );
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$memref ``
    custom<DynamicIndexList>($offsets, $static_offsets) `,`
    custom<DynamicIndexList>($lengths, $static_lengths) `,`
    custom<DynamicIndexList>($strides, $static_strides) `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($memref) `)` (`:` type($event)^)?
  }];
  let description = [{
    nd half dma operator.

    The access pattern is a 4-entry offset / length / stride triple, each
    entry being either a compile-time constant (stored in the corresponding
    `static_*` array) or an SSA `i64` value (stored in the matching variadic
    operand list, with `ShapedType::kDynamic` marking its slot in the static
    array). Use `getMixedOffsets()` / `getMixedLengths()` / `getMixedStrides()`
    to read an entry without caring which form it takes.
  }];
  let builders = [
    // Mixed static/dynamic builder; the preferred way to construct this op.
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "::mlir::Value":$id,
                   "::mlir::Value":$x,
                   "::mlir::Value":$y,
                   "::mlir::Value":$memref,
                   "::llvm::ArrayRef<::mlir::OpFoldResult>":$offsets,
                   "::llvm::ArrayRef<::mlir::OpFoldResult>":$lengths,
                   "::llvm::ArrayRef<::mlir::OpFoldResult>":$strides)>,
    // All-dynamic builder, for callers that already hold i64 Values.
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                   "::mlir::Value":$id,
                   "::mlir::Value":$x,
                   "::mlir::Value":$y,
                   "::mlir::Value":$memref,
                   "::mlir::ValueRange":$offsets,
                   "::mlir::ValueRange":$lengths,
                   "::mlir::ValueRange":$strides)>
  ];
  let extraClassDeclaration = [{
    /// Number of entries in each of the offset/length/stride lists.
    static constexpr unsigned kNumDims = 4;

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedOffsets() {
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(),
                                    getContext());
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedLengths() {
      return ::mlir::getMixedValues(getStaticLengths(), getLengths(),
                                    getContext());
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedStrides() {
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(),
                                    getContext());
    }

    /// Replace the offset/length/stride lists wholesale. Each list must have
    /// exactly `kNumDims` entries.
    void setMixedOffsets(::llvm::ArrayRef<::mlir::OpFoldResult> offsets);
    void setMixedLengths(::llvm::ArrayRef<::mlir::OpFoldResult> lengths);
    void setMixedStrides(::llvm::ArrayRef<::mlir::OpFoldResult> strides);
  }];
  let hasVerifier = 1;
}

def AIRRt_MemcpyNdOp: AIRRt_Op<"memcpy_nd", []> {
  let summary = "dma operator";
  let arguments = (
    ins AnyRankedOrUnrankedMemRef:$dst,
        AnyRankedOrUnrankedMemRef:$src,
        I64:$offset3,
        I64:$offset2,
        I64:$offset1,
        I64:$offset0,
        I64:$length3,
        I64:$length2,
        I64:$length1,
        I64:$length0,
        I64:$stride3,
        I64:$stride2,
        I64:$stride1,
        I64:$stride0
  );
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $dst`,`$src`,`
    `[` $offset3`,`$offset2`,`$offset1`,`$offset0 `]` `,`
    `[` $length3`,`$length2`,`$length1`,`$length0 `]` `,`
    `[` $stride3`,`$stride2`,`$stride1`,`$stride0 `]` `)` attr-dict `:`
    `(` type($dst)`,`type($src)`,`
    `[` type($offset3)`,`type($offset2)`,`type($offset1)`,` type($offset0) `]` `,`
    `[` type($length3)`,`type($length2)`,`type($length1)`,` type($length0) `]` `,`
    `[` type($stride3)`,`type($stride2)`,`type($stride1)`,` type($stride0) `]` `)`  (`:` type($event)^)?
  }];
  let description = [{
    nd memcpy operator
  }];
}

def AIRRt_AllocOp: AIRRt_Op<"alloc", []> {
  let summary = "AIRRt Allocation Op";
  let arguments = (ins);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
  let description = [{
    AIRRt Allocation Op
  }];
  let hasCanonicalizer  = 1;
}

def AIRRt_DeallocOp: AIRRt_Op<"dealloc", []> {
  let summary = "AIRRt Deallocation Op";
  let arguments = (ins AnyMemRef:$memref);
  let results = (outs);
  let assemblyFormat = [{
    attr-dict $memref `:` type($memref)
  }];
  let description = [{
    AIRRt Deallocation Op
  }];
  let hasCanonicalizer  = 1;
}

def AIRRt_WaitAllOp : AIRRt_Op<"wait_all", []> {
  let arguments = (ins Variadic<AIRRt_Event>:$events);
  let results = (outs Optional<AIRRt_Event>:$event);
  let summary = "wait for all operator";
  let description = [{
    Block and wait for all events before preceding.
    Optionally return an event.
  }];
  let assemblyFormat = [{
    ($events^)? attr-dict (`:` type($event)^)?
  }];
  let hasCanonicalizer  = 1;
}

#endif // #ifndef AIRRT_OPS
