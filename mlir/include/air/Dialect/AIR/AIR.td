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
    OperandRange getKernelOperands();
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
    
    /// Return the kernel argument that corresponds to the given operand.
    /// Return an "empty" block argument if the given value is not a kernel
    /// operand.
    BlockArgument getTiedKernelArgument(Value Oper) {
      auto kernelOperands = getKernelOperands();
      auto it = llvm::find(kernelOperands, Oper);
      if (it == kernelOperands.end())
        return {};
      return getKernelArgument(std::distance(kernelOperands.begin(), it));
    }
    /// Return the operand that corresponds to the given kernel argument.
    /// Return "nullptr" if the given block argument is not a kernel argument
    /// of this op.
    Value getTiedKernelOperand(BlockArgument bbArg) {
      auto args = getKernelArguments();
      auto it = llvm::find(args, bbArg);
      if (it == args.end())
        return {};
      return getKernelOperand(std::distance(args.begin(), it));
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
    OperandRange getKernelOperands();
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
    /// Return the kernel argument that corresponds to the given operand.
    /// Return an "empty" block argument if the given value is not a kernel
    /// operand.
    BlockArgument getTiedKernelArgument(Value Oper) {
      auto kernelOperands = getKernelOperands();
      auto it = llvm::find(kernelOperands, Oper);
      if (it == kernelOperands.end())
        return {};
      return getKernelArgument(std::distance(kernelOperands.begin(), it));
    }
    /// Return the operand that corresponds to the given kernel argument.
    /// Return "nullptr" if the given block argument is not a kernel argument
    /// of this op.
    Value getTiedKernelOperand(BlockArgument bbArg) {
      auto args = getKernelArguments();
      auto it = llvm::find(args, bbArg);
      if (it == args.end())
        return {};
      return getKernelOperand(std::distance(args.begin(), it));
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
                                       OptionalAttr<StrAttr>:$link_with,
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
    OperandRange getKernelOperands();
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

    /// Return the kernel argument that corresponds to the given operand.
    /// Return an "empty" block argument if the given value is not a kernel
    /// operand.
    BlockArgument getTiedKernelArgument(Value Oper) {
      auto kernelOperands = getKernelOperands();
      auto it = llvm::find(kernelOperands, Oper);
      if (it == kernelOperands.end())
        return {};
      return getKernelArgument(std::distance(kernelOperands.begin(), it));
    }
    /// Return the operand that corresponds to the given kernel argument.
    /// Return "nullptr" if the given block argument is not a kernel argument
    /// of this op.
    Value getTiedKernelOperand(BlockArgument bbArg) {
      auto args = getKernelArguments();
      auto it = llvm::find(args, bbArg);
      if (it == args.end())
        return {};
      return getKernelOperand(std::distance(args.begin(), it));
    }
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

def air_DmaMemcpyNdOp: air_Op<"dma_memcpy_nd",
                        [air_AsyncOpInterface,
                         air_MemcpyInterface,
                         AttrSizedOperandSegments]> {
  let summary = "dma operator";
  let arguments = (
    ins Variadic<air_AsyncToken>:$async_dependencies,
        AnyRankedOrUnrankedMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides,
        AnyRankedOrUnrankedMemRef:$src,
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
  let hasCanonicalizer = 1;
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

// AIR channel

def air_ChannelOp : air_Op<"channel", [Symbol]>,
    Arguments<(ins SymbolNameAttr:$sym_name,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$size,
                   DefaultValuedAttr<StrAttr, "\"dma_stream\"">:$channel_type)> {
  let assemblyFormat = [{
    $sym_name $size attr-dict
  }];
  let summary = "Channel for data movement.";
  let description = [{
    Operation to represent a communication channel as a point-to-point connection between two memrefs.
    The array following the channel name symbol represents the channel's dimensional sizes. Default
    size, with empty size array, is 1. The data movement mechanism that the channel uses is controlled 
    by the `channel_type` attribute.

    ### Channel Types
    The `channel_type` attribute is a string that determines the mechanism used for data movement:
    - **"dma_stream"** (default):  
      Use DMA engines to send and receive data, with routing performed over a streaming interconnect.
    - **"dma_packet"**:  
      Use DMA engines to send and receive data, with routing performed over a packet-switched network.
    - **"cascade"**:  
      Use processor cores to send and receive data via cascade connections between adjacent tiles.

    ### Broadcasting
    If a channel broadcasts to multiple destinations, the optional `broadcast_shape` attribute  
    annotates the output sizes after broadcasting. Broadcasting follows NumPy's broadcasting rules.

    Example:

    ```mlir
    // An array of 4 x 4 streaming DMA channels
    air.channel @channel_0 [4, 4] {channel_type = "dma_stream"}

    // A streaming DMA channel broadcasting to 4 destinations
    air.channel @channel_1 [1, 1] {broadcast_shape = [1, 4], channel_type = "dma_stream"}

    // An array of 1 x 4 streaming DMA channels broadcasting to 4 x 4 destinations.
    // Broadcasting follows NumPy's rules.
    air.channel @channel_2 [1, 4] {broadcast_shape = [4, 4], channel_type = "dma_stream"}

    // A packet-switched DMA channel
    air.channel @channel_3 [] {channel_type = "dma_packet"}

    // A cascade channel using core-to-core cascade connections
    air.channel @channel_4 [] {channel_type = "cascade"}
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
          auto attr = llvm::dyn_cast<IntegerAttr>(bShape).getInt();
          broadcastNum *= attr;
        }
      return broadcastNum;
    }
    int getBufferResources() {
      if(auto attr = getOperation()->getAttrOfType<IntegerAttr>("buffer_resources"))
        return llvm::dyn_cast<IntegerAttr>(attr).getInt();
      else
        return 1;
    }
    int getBundleSize() {
      int size = 1;
      for (auto i : getSize())
        size *= llvm::dyn_cast<IntegerAttr>(i).getInt();
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
        AnyRankedOrUnrankedMemRef:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides)>,
    Results<(outs Optional<air_AsyncToken>:$async_token)> {
  let summary = "Push for air channels.";
  let description = [{
    The `air.channel.put` operation represents a **push** (send) operation that copies data from a 
    source memref into a specified channel.

    This operation models one-way data movement into a channel endpoint, enabling asynchronous 
    communication between producer and consumer operations. It is typically paired with 
    `air.channel.get` operations on the receiving side.

    ### Semantics
    - The source data is specified by the `src` memref, along with its associated 
      `src_offsets`, `src_sizes`, and `src_strides` which describe the subview being transferred.
    - The channel being targeted is identified by the symbol referenced by `chan_name`.
    - The channel must have been declared earlier via an `air.channel` operation.
    - The operation may be asynchronous: if an async token is produced, it can be used to 
      synchronize with subsequent dependent operations.
    - The specific channel it operates on, when `chan_name` references an array of channels, is
      identified by `indices`.

    ### Interfaces
    - Implements `air_AsyncOpInterface`, allowing it to participate in async dependency chains.
    - Implements `air_MemcpyInterface`, enabling it to behave like a DMA/memcpy operation.
    - Implements `air_ChannelInterface`, allowing inspection of channel properties.

    ### Example

    ```mlir
    // Send a 4x4 tile from %src into channel @chan_0
    air.channel.put @chan_0(%src[%c0, %c0][%c4, %c4][%c1, %c1]) : (memref<16x16xf32>)
    
    // Asynchronous put with dependency on %t0
    %t1 = air.channel.put async [%t0] @chan_1(%src[%c8, %c0][%c4, %c4][%c1, %c1]) : (memref<16x16xf32>)
    ```
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
  let hasCanonicalizer = 1;
}

def air_ChannelGetOp : air_Op<"channel.get", [air_AsyncOpInterface, 
                                              air_MemcpyInterface, 
                                              air_ChannelInterface, 
                                              AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<air_AsyncToken>:$async_dependencies,
        FlatSymbolRefAttr:$chan_name,
        Variadic<Index>:$indices,
        AnyRankedOrUnrankedMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides)>,
    Results<(outs Optional<air_AsyncToken>:$async_token)> {
  let summary = "Get for air channels.";
  let description = [{
    The `air.channel.get` operation represents a **pull** (receive) operation that copies data from a 
    specified channel into a destination memref.

    This operation models one-way data movement from a channel endpoint into memory, enabling 
    asynchronous communication where data previously sent by a corresponding 
    `air.channel.put` becomes available to the consumer.

    ### Semantics
    - The destination buffer is specified by the `dst` memref, along with its associated 
      `dst_offsets`, `dst_sizes`, and `dst_strides` which describe the subview being written to.
    - The channel being read is identified by the symbol referenced by `chan_name`.
    - The channel must have been declared earlier via an `air.channel` operation.
    - The operation may be asynchronous: if an async token is produced, it can be used to 
      synchronize with subsequent dependent operations.
    - The specific channel it operates on, when `chan_name` references an array of channels, is
      identified by `indices`.

    ### Interfaces
    - Implements `air_AsyncOpInterface`, enabling participation in async dependency chains.
    - Implements `air_MemcpyInterface`, allowing it to behave like a DMA/memcpy operation.
    - Implements `air_ChannelInterface`, allowing inspection of channel properties.

    ### Example

    ```mlir
    // Receive a 4x4 tile into %dst from channel @chan_0
    air.channel.get @chan_0(%dst[%c0, %c0][%c4, %c4][%c1, %c1]) : (memref<16x16xf32>)
    
    // Asynchronous get with dependency on %t1
    %t2 = air.channel.get async [%t1] @chan_1(%dst[%c8, %c0][%c4, %c4][%c1, %c1]) : (memref<16x16xf32>)
    ```
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
  let hasCanonicalizer = 1;
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
  let regions = (region SizedRegion<1>:$region);
  let description = [{
    Defines a code region to be dispatched asynchronously at runtime. All operations in
    the region must be executed sequentially.
  }];

  // Note: async_dependencies are printed in sorted order for ops which have
  // custom assembly formats. We can consider giving ExecuteOp custom
  // printer/parser too.
  let assemblyFormat = [{
    (` ``[` $async_dependencies^ `]`)?
    (`->` `(` type($results)^ `)`)? regions attr-dict
  }];

  let extraClassDeclaration = [{
    Block &getBody() { return getRegion().front(); }
    llvm::iplist<Operation> &getChildOps() { return getBody().getOperations(); }
    SmallVector<Operation *> getYieldedChildOps() {
      SmallVector<Operation *> ops;
      for (auto oper : getBody().getTerminator()->getOperands())
        if (oper.getDefiningOp() && getRegion().isAncestor(oper.getDefiningOp()->getParentRegion()))
          ops.push_back(oper.getDefiningOp());
      return ops;
    }
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
