//===- AIR.td ----------------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_OPS
#define AIR_OPS

include "mlir/IR/OpBase.td"
include "air/Dialect/AIR/AIROpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class air_Op<string mnemonic, list<Trait> traits = []> :
    Op<air_Dialect, mnemonic, traits>;

def air_LaunchOp : air_Op<"launch", [air_AsyncOpInterface,
                                     air_HierarchyInterface,
                                     AttrSizedOperandSegments,
                                     IsolatedFromAbove,
                                     AffineScope,
                                     SingleBlockImplicitTerminator<"LaunchTerminatorOp">]>,
                          Arguments<(ins OptionalAttr<SymbolNameAttr>:$sym_name,
                                         Variadic<air_AsyncToken>:$async_dependencies,
                                         Variadic<Index>:$sizes,
                                         Variadic<AnyType>:$launch_operands)>,
                          Results<(outs Optional<air_AsyncToken>:$async_token)> {
  let summary = "Launch";
  let description = [{
    Launch
  }];

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$sizes,"ValueRange":$launch_operands)>,
    OpBuilder<(ins "ValueRange":$async_dependencies,
      "ValueRange":$sizes,"ValueRange":$launch_operands,
      CArg<"bool", "false">:$is_async,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    ArrayRef<BlockArgument> getIds();
    ArrayRef<BlockArgument> getSize();
    OperandRange getSizeOperands();
    unsigned getNumKernelOperands();
    Value getKernelOperand(unsigned i);
    ArrayRef<BlockArgument> getKernelArguments();
    BlockArgument getKernelArgument(unsigned i);
    unsigned getNumDims();
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
  let hasCanonicalizer = 1;
}

def air_LaunchTerminatorOp : air_Op<"launch_terminator", [HasParent<"LaunchOp">,
                                                          Pure,
                                                          Terminator]>,
                                    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for `air.launch`.";
  let description = [{
    A terminator operation for the body of `air.launch` operations.
    `air.launch` operations are not expected to return any value so the
    terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}

def air_SegmentOp : air_Op<"segment", [air_AsyncOpInterface,
                                           air_HierarchyInterface,
                                           AttrSizedOperandSegments,
                                           IsolatedFromAbove,
                                           AffineScope,
                                           SingleBlockImplicitTerminator<"SegmentTerminatorOp">]>,
                        Arguments<(ins OptionalAttr<SymbolNameAttr>:$sym_name,
                                       Variadic<air_AsyncToken>:$async_dependencies,
                                       Variadic<Index>:$sizes,
                                       Variadic<AnyType>:$segment_operands)>,
                        Results<(outs Optional<air_AsyncToken>:$async_token)> {
  let summary = "Segment";
  let description = [{
    Segment
  }];

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$sizes,"ValueRange":$segment_operands)>,
    OpBuilder<(ins "ValueRange":$async_dependencies,
      "ValueRange":$sizes,"ValueRange":$segment_operands,
      CArg<"bool", "false">:$is_async,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    ArrayRef<BlockArgument> getIds();
    ArrayRef<BlockArgument> getSize();
    OperandRange getSizeOperands();
    unsigned getNumKernelOperands();
    Value getKernelOperand(unsigned i);
    ArrayRef<BlockArgument> getKernelArguments();
    BlockArgument getKernelArgument(unsigned i);
    unsigned getNumDims();
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
    static StringRef getColOffsetAttrName() { return "x_loc"; }
    static StringRef getRowOffsetAttrName() { return "y_loc"; }
    std::optional<uint64_t> getColOffset() {
      auto name = getColOffsetAttrName();
      if (auto a = (*this)->getAttrOfType<IntegerAttr>(name)) {
        return a.getInt();
      }
      return std::optional<uint64_t>();
    }
    std::optional<uint64_t> getRowOffset() {
      auto name = getRowOffsetAttrName();
      if (auto a = (*this)->getAttrOfType<IntegerAttr>(name)) {
        return a.getInt();
      }
      return std::optional<uint64_t>();
    }
    static StringRef getNumColsAttrName() { return "x_size"; }
    static StringRef getNumRowsAttrName() { return "y_size"; }
    std::optional<uint64_t> getNumCols() {
      auto name = getNumColsAttrName();
      if (auto a = (*this)->getAttrOfType<IntegerAttr>(name)) {
        return a.getInt();
      }
      return std::optional<uint64_t>();
    }
    std::optional<uint64_t> getNumRows() {
      auto name = getNumRowsAttrName();
      if (auto a = (*this)->getAttrOfType<IntegerAttr>(name)) {
        return a.getInt();
      }
      return std::optional<uint64_t>();
    }
  }];
  let hasCanonicalizer = 1;
}

def air_SegmentTerminatorOp : air_Op<"segment_terminator", [HasParent<"SegmentOp">,
                                                      Pure, Terminator]>,
    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for air segment regions.";
  let description = [{
    A terminator operation for the body of `air.segment` operations.
    `air.segment` operations are not expected to return any value so the
    terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}

def air_HerdOp : air_Op<"herd", [air_AsyncOpInterface,
                                 air_HierarchyInterface,
                                 AttrSizedOperandSegments,
                                 IsolatedFromAbove,
                                 AffineScope,
                                 SingleBlockImplicitTerminator<"HerdTerminatorOp">]>,
                        Arguments<(ins OptionalAttr<SymbolNameAttr>:$sym_name,
                                       Variadic<air_AsyncToken>:$async_dependencies,
                                       Variadic<Index>:$sizes,
                                       Variadic<AnyType>:$herd_operands)>,
                        Results<(outs Optional<air_AsyncToken>:$async_token)> {
  let summary = "Herd";
  let description = [{
    Define and run a 1D or 2D array of tiles as an AIR Herd.
  }];

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$sizes,"ValueRange":$herd_operands)>,
    OpBuilder<(ins "ValueRange":$async_dependencies,
      "ValueRange":$sizes,
      "ValueRange":$herd_operands,
      CArg<"bool", "false">:$is_async,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    ArrayRef<BlockArgument> getIds();
    ArrayRef<BlockArgument> getSize();
    OperandRange getSizeOperands();
    unsigned getNumKernelOperands();
    Value getKernelOperand(unsigned i);
    ArrayRef<BlockArgument> getKernelArguments();
    BlockArgument getKernelArgument(unsigned i);
    unsigned getNumDims();
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
    static StringRef getColOffsetAttrName() { return "x_loc"; }
    static StringRef getRowOffsetAttrName() { return "y_loc"; }
    std::optional<uint64_t> getColOffset() {
      auto name = getColOffsetAttrName();
      if (auto a = (*this)->getAttrOfType<IntegerAttr>(name)) {
        return a.getInt();
      }
      return std::optional<uint64_t>();
    }
    std::optional<uint64_t> getRowOffset() {
      auto name = getRowOffsetAttrName();
      if (auto a = (*this)->getAttrOfType<IntegerAttr>(name)) {
        return a.getInt();
      }
      return std::optional<uint64_t>();
    }
    uint64_t getNumCols();
    uint64_t getNumRows();
  }];
  let hasCanonicalizer = 1;
}

def air_HerdTerminatorOp : air_Op<"herd_terminator", [HasParent<"HerdOp">,
                                                      Pure, Terminator]>,
                                   Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for air herd regions.";
  let description = [{
    A terminator operation for the body of `air.herd` operations.
    `air.herd` operations are not expected to return any value so the
    terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}

def air_HerdPipelineOp : air_Op<"pipeline", [HasParent<"HerdOp">,
                                             AffineScope]>,
                          Arguments<(ins)>,
                          Results<(outs)> {
  let summary = "Define a pipeline";
  let description = [{
    Define a pipeline within an AIR Herd.
  }];

  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict-with-keyword $body
  }];
  let extraClassDeclaration = [{
    // Return a vector of the air.pipeline.stage ops for this pipeline
    SmallVector<PipelineStageOp,8> getStages();
  }];
}

def air_PipelineStageOp : air_Op<"pipeline.stage", [HasParent<"HerdPipelineOp">]>,
    Arguments<(ins Variadic<AnyType>:$opers)>, Results<(outs Variadic<AnyType>:$results)> {
  let summary = "Pipeline stage";
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Pipeline stage.
  }];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    // Return the offset of this stage in the pipeline"
    unsigned getStageId();
  }];
}

def air_PipelineYieldOp : air_Op<"pipeline.yield", [HasParent<"PipelineStageOp">,
                                                    Pure, ReturnLike, Terminator]>,
                                 Arguments<(ins Variadic<AnyType>:$opers)>, Results<(outs)> {
  let summary = "Yield for air pipeline stages.";
  let description = [{
    A terminator operation for regions that appear in the body of
    `air.pipeline.stage` operation. The operation takes variable number of
    operands and produces no results. The operand number and types must
    match the signature of the `air.pipeline` that contains the operation.
  }];
  let assemblyFormat = [{
    ($opers^)? attr-dict (`:` type($opers)^)?
  }];
}

def air_PipelinePutOp : air_Op<"pipeline.put", []>,
    Arguments<(ins AnyType:$dst0, AnyType:$dst1, Variadic<AnyType>:$opers)>, Results<(outs)> {
  let summary = "Put for air pipeline stages.";
  let description = [{
    Experimental operation to represent copying data to another tile.
    Currently used internally by air-to-aie pass during pipeline lowering.
  }];
  let assemblyFormat = [{
    $dst0 $dst1 (`,` $opers^)? attr-dict `:` type($dst0) `,` type($dst1) (`,` type($opers)^)?
  }];
}

def air_PipelineGetOp : air_Op<"pipeline.get", []>,
    Arguments<(ins AnyType:$src0, AnyType:$src1)>, Results<(outs Variadic<AnyType>:$results)> {
  let summary = "Get for air pipeline stages.";
  let description = [{
    Experimental operation to represent copying data from another tile.
    Currently used internally by air-to-aie pass during pipeline lowering.
  }];
  let assemblyFormat = [{
    $src0 $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($results)
  }];
}

def air_PipelineTerminatorOp : air_Op<"pipeline.terminator", [HasParent<"HerdPipelineOp">,
                                          Pure, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$opers)>, Results<(outs)> {
  let summary = "Terminator for air pipeline regions.";
  let description = [{
    A terminator operation for regions that appear in the body of
    `air.pipeline` operation.
  }];
  let assemblyFormat = "attr-dict ($opers^ `:` type($opers))?";
}

def air_DmaMemcpyNdOp: air_Op<"dma_memcpy_nd",
                        [air_AsyncOpInterface,
                         air_MemcpyInterface,
                         AttrSizedOperandSegments]> {
  let summary = "dma operator";
  let arguments = (
    ins Variadic<air_AsyncToken>:$async_dependencies,
        AnyMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides,
        AnyMemRef:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides
  );
  let results = (outs Optional<air_AsyncToken>:$async_token);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($async_token), $async_dependencies)
    `(` $dst `[` ($dst_offsets^)? `]``[` ($dst_sizes^)? `]``[` ($dst_strides^)? `]` `,`
        $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)`  attr-dict `:`
    `(` type($dst) `,` type($src) `)`
  }];
  let description = [{
    dma operator
  }];
  let extraClassDeclaration = [{
    Value getSrcMemref() { return getSrc(); }
    Value getDstMemref() { return getDst(); }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

def air_WaitAllOp: air_Op<"wait_all", [air_AsyncOpInterface]> {
  let arguments = (ins Variadic<air_AsyncToken>:$async_dependencies);
  let results = (
    outs Optional<air_AsyncToken>:$async_token
  );

  let summary = "wait for all operator";
  let description = [{
    Wait for all async tokens before preceding.
  }];
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($async_token), $async_dependencies) attr-dict
  }];
  let extraClassDeclaration = [{
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
  let hasCanonicalizer = 1;
}

def air_AllocOp: air_Op<"alloc", [air_AsyncOpInterface]> {
  let summary = "alloc operator";
  let arguments = (
    ins Variadic<air_AsyncToken>:$async_dependencies
  );
  let results = (outs Optional<air_AsyncToken>:$async_token, AnyMemRef:$result);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($async_token), $async_dependencies) attr-dict `:` type($result)
  }];
  let description = [{
    Allocate memory
  }];
}

def air_DeallocOp: air_Op<"dealloc", [air_AsyncOpInterface]> {
  let summary = "dealloc operator";
  let arguments = (
    ins Variadic<air_AsyncToken>:$async_dependencies, AnyMemRef:$memref
  );
  let results = (outs Optional<air_AsyncToken>:$async_token);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($async_token), $async_dependencies) $memref attr-dict `:` type($memref)
  }];
  let description = [{
    Deallocate memory
  }];
}

// AIR channel

def air_ChannelOp : air_Op<"channel", [Symbol]>,
    Arguments<(ins SymbolNameAttr:$sym_name,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$size)> {
  let assemblyFormat = [{
    $sym_name $size attr-dict
  }];
  let summary = "Channel for data movement.";
  let description = [{
    Operation to represent a channel as a point-to-point connection between two memrefs. The array
    following the channel name symbol represents the channel sizes. If each channel is broadcasting
    to multiple destinations, then the optional 'broadcast_shape' attribute annotates the output 
    sizes after broadcasting.

    Example:

    ```mlir
    air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
    ```
  }];
  let extraClassDeclaration = [{
    bool isBroadcast() {
      if(auto attr = getOperation()->getAttrOfType<ArrayAttr>("broadcast_shape"))
        return true;
      else
        return false;
    }
    int getBroadcastDimension();
    ArrayAttr getBroadcastShape() {
      return getOperation()->getAttrOfType<ArrayAttr>("broadcast_shape");
    }
    int getBroadcastNum() {
      int broadcastNum = 1;
      if (isBroadcast())
        for (auto bShape : getOperation()->getAttrOfType<ArrayAttr>("broadcast_shape")) {
          auto attr = bShape.dyn_cast<IntegerAttr>().getInt();
          broadcastNum *= attr;
        }
      return broadcastNum;
    }
    int getBufferResources() {
      if(auto attr = getOperation()->getAttrOfType<IntegerAttr>("buffer_resources"))
        return attr.dyn_cast<IntegerAttr>().getInt();
      else
        return 1;
    }
    int getBundleSize() {
      int size = 1;
      for (auto i : getSize())
        size *= i.dyn_cast<IntegerAttr>().getInt();
      return size;
    }
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def air_ChannelPutOp : air_Op<"channel.put", [air_AsyncOpInterface, 
                                              air_MemcpyInterface, 
                                              air_ChannelInterface, 
                                              AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<air_AsyncToken>:$async_dependencies,
        FlatSymbolRefAttr:$chan_name,
        Variadic<Index>:$indices,
        AnyMemRef:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides)>,
    Results<(outs Optional<air_AsyncToken>:$async_token)> {
  let summary = "Push for air channels.";
  let description = [{
    Experimental operation to represent copying data from a memref to a channel.
  }];
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($async_token), $async_dependencies)
    $chan_name `[` ($indices^)? `]`
    `(` $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)` attr-dict `:`
    `(` type($src) `)`
  }];
  let extraClassDeclaration = [{
    Value getMemref() { return getSrc(); }
    Value getSrcMemref() { return getSrc(); }
    Value getDstMemref() { return nullptr; }
    OperandRange getOffsets() { return getSrcOffsets(); }
    OperandRange getDstOffsets() { emitOpError("does not have DstOffset"); llvm_unreachable("unreachable"); }
    OperandRange getSizes() { return getSrcSizes(); }
    OperandRange getDstSizes() { emitOpError("does not have DstSizes"); llvm_unreachable("unreachable"); }
    OperandRange getStrides() { return getSrcStrides(); }
    OperandRange getDstStrides() { emitOpError("does not have DstStrides"); llvm_unreachable("unreachable"); }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

def air_ChannelGetOp : air_Op<"channel.get", [air_AsyncOpInterface, 
                                              air_MemcpyInterface, 
                                              air_ChannelInterface, 
                                              AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<air_AsyncToken>:$async_dependencies,
        FlatSymbolRefAttr:$chan_name,
        Variadic<Index>:$indices,
        AnyMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides)>,
    Results<(outs Optional<air_AsyncToken>:$async_token)> {
  let summary = "Get for air channels.";
  let description = [{
    Experimental operation to represent copying from a channel to a memref.
  }];
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($async_token), $async_dependencies)
    $chan_name `[` ($indices^)? `]`
    `(` $dst `[` ($dst_offsets^)? `]``[` ($dst_sizes^)? `]``[` ($dst_strides^)? `]` `)` attr-dict `:`
    `(` type($dst) `)`
  }];
  let extraClassDeclaration = [{
    Value getMemref() { return getDst(); }
    Value getDstMemref() { return getDst(); }
    Value getSrcMemref() { return nullptr; }
    OperandRange getOffsets() { return getDstOffsets(); }
    OperandRange getSrcOffsets() { emitOpError("does not have SrcOffsets"); llvm_unreachable("unreachable"); }
    OperandRange getSizes() { return getDstSizes(); }
    OperandRange getSrcSizes() { emitOpError("does not have SrcSizes"); llvm_unreachable("unreachable"); }
    OperandRange getStrides() { return getDstStrides(); }
    OperandRange getSrcStrides() { emitOpError("does not have SrcStrides"); llvm_unreachable("unreachable"); }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

// AIR asynchronous region for dynamic event dispatching.

def air_ExecuteOp : air_Op<"execute", [SingleBlockImplicitTerminator<"ExecuteTerminatorOp">,
                                       air_AsyncOpInterface]> {
  let arguments = (
    ins Variadic<air_AsyncToken>:$async_dependencies
  );
  let results = (
    outs air_AsyncToken:$async_token,
         Variadic<AnyType>:$results
  );
  let summary = "Asynchronous code region";
  let regions = (region SizedRegion<1>:$body);
  let description = [{
    Defines a code region to be dispatched asynchronously at runtime. All operations in
    the region must be executed sequentially.
  }];

  // TODO(jn) async_dependencies are printed in sorted order for ops which have
  // custom assembly formats. ExecuteOp should be made to have a custom print
  // and parse too, so that it follows this pattern.
  let assemblyFormat = [{
    (` ``[` $async_dependencies^ `]`)?
    (`->` `(` type($results)^ `)`)? regions attr-dict
  }];

  let extraClassDeclaration = [{
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def air_ExecuteTerminatorOp : air_Op<"execute_terminator", [HasParent<"ExecuteOp">,
                                                      Pure, ReturnLike, Terminator]>{
  let summary = "Terminator for air execute.";
  let description = [{
    A terminator operation for code regions that appear in the body of
    `air.execute` operation. The operation takes variable number of
    operands and produces no results. The operand number and types must
    match the signature of the `air.execute` that contains the operation.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

// AIR custom op, as a handle for a user-provided AIE kernel

def air_CustomOp : air_Op<"custom", [air_AsyncOpInterface,
                                 AttrSizedOperandSegments]>,
                        Arguments<(ins OptionalAttr<SymbolRefAttr>:$symbol,
                                       Variadic<air_AsyncToken>:$async_dependencies,
                                       Variadic<AnyType>:$custom_operands)>,
                        Results<(outs Optional<air_AsyncToken>:$async_token)> {
  let summary = "A handle to a user-customized op";
  let description = [{
    A placeholder operation for a user-customized op. With user-specified 
    latency value, AIR Runner is able to simulate the system-level
    performance with this op in place.
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$custom_operands)>,
    OpBuilder<(ins "ValueRange":$async_dependencies,
      "ValueRange":$custom_operands,
      CArg<"bool", "false">:$is_async,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];
  let hasCustomAssemblyFormat = 1;
}

#endif
