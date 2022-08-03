// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.


#ifndef AIR_OPS
#define AIR_OPS

include "mlir/IR/OpBase.td"
include "air/Dialect/AIR/AIROpBase.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class air_Op<string mnemonic, list<Trait> traits = []> :
    Op<air_Dialect, mnemonic, traits>;

def air_HerdLaunchOp : air_Op<"launch_herd", [air_AsyncOpInterface,
                                              AttrSizedOperandSegments,
                                              IsolatedFromAbove,
                                              AffineScope]>,
                        Arguments<(ins Variadic<air_AsyncToken>:$asyncDependencies,
                                       Index:$herdSizeX, Index:$herdSizeY,
                                       Variadic<AnyType>:$operands)>,
                        Results<(outs Optional<air_AsyncToken>:$asyncToken)> {
  let summary = "Launch a herd";
  let description = [{
    Define and run a 1D or 2D array of tiles as an AIR Herd.
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "HerdDim2":$gridSize,"ValueRange":$kernelOperands)>,
    OpBuilder<(ins "ValueRange":$asyncDependencies,
      "HerdDim2":$gridSize,
      "ValueRange":$kernelOperands,
      CArg<"bool", "false">:$isAsync
    )>
  ];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    HerdDim2 getTileIds();
    HerdDim2 getHerdSize();
    HerdDim2 getHerdSizeOperands();
    unsigned getNumKernelOperands();
    Value getKernelOperand(unsigned i);
    ArrayRef<BlockArgument> getKernelArguments();
    BlockArgument getKernelArgument(unsigned i);
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
  let hasCanonicalizer  = 1;
}

def air_HerdTerminatorOp : air_Op<"herd_terminator", [HasParent<"HerdLaunchOp">,
                                                      NoSideEffect, Terminator]>,
    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for air launch_herd regions.";
  let description = [{
    A terminator operation for regions that appear in the body of
    `air.launch_herd` operation.  These regions are not expected to return any
    value so the terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}

def air_HerdPipelineOp : air_Op<"pipeline", [HasParent<"HerdLaunchOp">,
                                             AffineScope]>,
                          Arguments<(ins)>,
                          Results<(outs)> {
  let summary = "Define a pipeline";
  let description = [{
    Define a pipeline within an AIR Herd.
  }];

  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict $body
  }];
  let extraClassDeclaration = [{
    // Return a vector of the air.pipeline.stage ops for this pipeline
    SmallVector<PipelineStageOp,8> getStages();
  }];
}

def air_PipelineStageOp : air_Op<"pipeline.stage", [HasParent<"HerdPipelineOp">]>,
    Arguments<(ins Variadic<AnyType>:$opers)>, Results<(outs Variadic<AnyType>:$results)> {
  let summary = "Pipeline stage";
  let regions = (region AnyRegion:$body);
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
                                          NoSideEffect, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$opers)>, Results<(outs)> {
  let summary = "Yield for air pipeline stages.";
  let description = [{
    A terminator operation for regions that appear in the body of 
    `air.pipeline.stage` operation.
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
                                          NoSideEffect, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$opers)>, Results<(outs)> {
  let summary = "Terminator for air pipeline regions.";
  let description = [{
    A terminator operation for regions that appear in the body of 
    `air.pipeline` operation.
  }];
  let assemblyFormat = "attr-dict ($opers^ `:` type($opers))?";
}

def air_DmaMemcpy4dOp: air_Op<"dma_memcpy_4d", [air_AsyncOpInterface, air_DmaMemcpyInterface]> {
  let summary = "dma operator";
  let arguments = (
    ins Variadic<air_AsyncToken>:$asyncDependencies,
        AnyMemRef:$dst,
        AnyMemRef:$src,
        Index:$dst_d3,
        Index:$dst_d2,
        Index:$dst_d1,
        Index:$dst_d0,
        Index:$src_d3,
        Index:$src_d2,
        Index:$src_d1,
        Index:$src_d0,
        Index:$num,
        Index:$stride,
        Index:$elem_per_stride
  );
  let results = (outs Optional<air_AsyncToken>:$asyncToken);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) 
    `(`$dst`,` $src`,`
    `[` $dst_d3`,`$dst_d2`,`$dst_d1`,`$dst_d0 `]` `,`
    `[` $src_d3`,`$src_d2`,`$src_d1`,`$src_d0 `]` `,`
    $num `,` $stride `,` $elem_per_stride `)` attr-dict `:`
    `(` type($dst)`,` type($src)`,`
    `[` type($dst_d3)`,`type($dst_d2)`,`type($dst_d1)`,` type($dst_d0) `]` `,`
    `[` type($src_d3)`,`type($src_d2)`,`type($src_d1)`,` type($src_d0) `]` `,`
    type($num) `,` type($stride) `,` type($elem_per_stride) `)`  `->` `(` `)`
  }];
  let description = [{
    dma operator
  }];
  let extraClassDeclaration = [{
    unsigned getNumDims() { return 4; }
    Value getSrcMemref() { return src(); }
    Value getDstMemref() { return dst(); }
    Value getSrcMemrefDim(unsigned i) { 
      return SmallVector<Value, 4> {src_d3(), src_d2(), src_d1(), src_d0()}[i]; 
    }
    Value getDstMemrefDim(unsigned i) { 
      return SmallVector<Value, 4> {dst_d3(), dst_d2(), dst_d1(), dst_d0()}[i]; 
    }
    Value getLength() { return num(); }
    Value getStride() { return stride(); }
    Value getElemPerStride() { return elem_per_stride(); }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

def air_DmaMemcpy2dOp: air_Op<"dma_memcpy_2d", [air_AsyncOpInterface, air_DmaMemcpyInterface]> {
  let summary = "dma operator";
  let arguments = (
    ins Variadic<air_AsyncToken>:$asyncDependencies,
        AnyMemRef:$dst,
        AnyMemRef:$src,
        Index:$dst_d1,
        Index:$dst_d0,
        Index:$src_d1,
        Index:$src_d0,
        Index:$num,
        Index:$stride,
        Index:$elem_per_stride
  );
  let results = (outs Optional<air_AsyncToken>:$asyncToken);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) 
    `(`$dst`,` $src`,`
    `[` $dst_d1`,` $dst_d0 `]` `,`
    `[` $src_d1`,` $src_d0 `]` `,`
    $num `,` $stride `,` $elem_per_stride `)` attr-dict `:`
    `(` type($dst)`,` type($src)`,`
    `[` type($dst_d1)`,` type($dst_d0) `]` `,`
    `[` type($src_d1)`,` type($src_d0) `]` `,`
    type($num) `,` type($stride) `,` type($elem_per_stride) `)`  `->` `(` `)`
  }];
  let description = [{
    dma operator
  }];
  let extraClassDeclaration = [{
    unsigned getNumDims() { return 2; }
    Value getSrcMemref() { return src(); }
    Value getDstMemref() { return dst(); }
    Value getSrcMemrefDim(unsigned i) { 
      return SmallVector<Value, 2> {src_d1(), src_d0()}[i]; 
    }
    Value getDstMemrefDim(unsigned i) { 
      return SmallVector<Value, 2> {dst_d1(), dst_d0()}[i]; 
    }
    Value getLength() { return num(); }
    Value getStride() { return stride(); }
    Value getElemPerStride() { return elem_per_stride(); }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

def air_DmaMemcpyOp: air_Op<"dma_memcpy", [air_AsyncOpInterface, air_DmaMemcpyInterface]> {
  let summary = "dma operator";
  let arguments = (
    ins Variadic<air_AsyncToken>:$asyncDependencies,
        AnyMemRef:$dst,
        AnyMemRef:$src,
        Index:$dst_d0,
        Index:$src_d0,
        Index:$num
  );
  let results = (outs Optional<air_AsyncToken>:$asyncToken);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) 
    `(`$dst`,` $src`,`
    `[` $dst_d0 `]` `,`
    `[` $src_d0 `]` `,`
    $num `)`  attr-dict `:`
    `(` type($dst)`,` type($src)`,`
    `[` type($dst_d0) `]` `,`
    `[` type($src_d0) `]` `,`
    type($num) `)`  `->` `(` `)`
  }];
  let description = [{
    dma operator
  }];
  let extraClassDeclaration = [{
    unsigned getNumDims() { return 1; }
    Value getSrcMemref() { return src(); }
    Value getDstMemref() { return dst(); }
    Value getSrcMemrefDim(unsigned i) { return src_d0(); }
    Value getDstMemrefDim(unsigned i) { return dst_d0(); }
    Value getLength() { return num(); }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

def air_DmaMemcpyNdOp: air_Op<"dma_memcpy_nd",
                        [air_AsyncOpInterface,
                         air_DmaMemcpyInterface,
                         AttrSizedOperandSegments]> {
  let summary = "dma operator";
  let arguments = (
    ins Variadic<air_AsyncToken>:$asyncDependencies,
        AnyMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides,
        AnyMemRef:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides
  );
  let results = (outs Optional<air_AsyncToken>:$asyncToken);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) 
    `(` $dst `[` ($dst_offsets^)? `]``[` ($dst_sizes^)? `]``[` ($dst_strides^)? `]` `,` 
        $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)`  attr-dict `:`
    `(` type($dst) `,` type($src) `)`
  }];
  let description = [{
    dma operator
  }];
  let extraClassDeclaration = [{
    unsigned getNumDims() { return 0; }
    Value getSrcMemref() { return src(); }
    Value getDstMemref() { return dst(); }
    SmallVector<Value, 4> getDstOffsets(){ return dst_offsets(); }
    SmallVector<Value, 4> getSrcOffsets(){ return src_offsets(); }
    SmallVector<Value, 4> getDstSizes(){ return dst_sizes(); }
    SmallVector<Value, 4> getSrcSizes(){ return src_sizes(); }
    SmallVector<Value, 4> getDstStrides(){ return dst_strides(); }
    SmallVector<Value, 4> getSrcStrides(){ return src_strides(); }
    Value getSrcMemrefDim(unsigned i) { return nullptr; }
    Value getDstMemrefDim(unsigned i) { return nullptr; }
    Value getLength() { return nullptr; }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

def air_WaitAllOp: air_Op<"wait_all", [air_AsyncOpInterface]> {
  let arguments = (ins Variadic<air_AsyncToken>:$asyncDependencies);
  let results = (
    outs Optional<air_AsyncToken>:$asyncToken
  );

  let summary = "wait for all operator";
  let description = [{
    Wait for all async tokens before preceding.
  }];
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
  }];
  let extraClassDeclaration = [{
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

// AIR channel

def air_ChannelOp : air_Op<"channel", []>,
    Arguments<(ins FlatSymbolRefAttr:$name, 
        Confined<I32Attr, [IntMinValue<0>]>:$countx,
        Confined<I32Attr, [IntMinValue<0>]>:$county)> {
  let assemblyFormat = [{
    $name `{` `count_x` `=` $countx `,` `count_y` `=` $county `}` attr-dict 
  }];
  let summary = "Channel for data movement.";
  let description = [{
    Experimental operation to represent a channel as a point-to-point connection between
    two memrefs.
  }];
}

def air_ForkOp : air_Op<"fork", []>,
    Arguments<(ins FlatSymbolRefAttr:$chan_in_name, 
        FlatSymbolRefAttr:$chan_out_name)> {
  let assemblyFormat = [{
    $chan_in_name $chan_out_name attr-dict 
  }];
  let summary = "Forked channel for broadcast data movement.";
  let description = [{
    Experimental operation to represent a channel fork creating a data broadcast.
  }];
}

def air_PushOp : air_Op<"push", [air_AsyncOpInterface, AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<air_AsyncToken>:$asyncDependencies, 
        FlatSymbolRefAttr:$chan_name, 
        Index:$chan_idx,
        Index:$chan_idy,
        AnyMemRef:$src,
        Variadic<Index>:$src_offsets,
        Variadic<Index>:$src_sizes,
        Variadic<Index>:$src_strides)>, 
    Results<(outs Optional<air_AsyncToken>:$asyncToken)> {
  let summary = "Push for air channels.";
  let description = [{
    Experimental operation to represent copying data into a channel.
  }];
  let assemblyFormat = [{
    $chan_name `(` $chan_idx `,` $chan_idy `)`
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) 
    `(` $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)`  attr-dict `:`
    `(` type($src) `)`
  }];
}

def air_PopOp : air_Op<"pop", [air_AsyncOpInterface, AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<air_AsyncToken>:$asyncDependencies, 
        FlatSymbolRefAttr:$chan_name, 
        Index:$chan_idx,
        Index:$chan_idy,
        AnyMemRef:$dst,
        Variadic<Index>:$dst_offsets,
        Variadic<Index>:$dst_sizes,
        Variadic<Index>:$dst_strides)>, 
    Results<(outs Optional<air_AsyncToken>:$asyncToken)> {
  let summary = "Pop for air channels.";
  let description = [{
    Experimental operation to represent copying data out of a channel.
  }];
  let assemblyFormat = [{
    $chan_name `(` $chan_idx `,` $chan_idy `)`
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) 
    `(` $dst `[` ($dst_offsets^)? `]``[` ($dst_sizes^)? `]``[` ($dst_strides^)? `]` `)`  attr-dict `:`
    `(` type($dst) `)`
  }];
}

// AIR asynchronous region for dynamic event dispatching.

def air_RegionOp : air_Op<"region", [air_AsyncOpInterface, AffineScope]> {
  let arguments = (
    ins Variadic<AnyType>:$asyncDependencies
  );
  let results = (
    outs  air_AsyncToken:$asyncToken,
          Optional<AnyType>:$valOut
  );
  let summary = "Asynchronous code region";
  let regions = (region AnyRegion:$body);
  let description = [{
    Defines a code region to be dispatched asynchronously at runtime. All operations in
    the region must be executed sequentially.
  }];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) (`:` 
    `(` type($asyncDependencies)^ `)`)? regions attr-dict (`:` `(` type($valOut)^ `)`)?
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
}

def air_RegionTerminatorOp : air_Op<"region_terminator", [HasParent<"RegionOp">,
                                                      NoSideEffect, ReturnLike, Terminator]>{
  let summary = "Terminator for air regions.";
  let description = [{
    A terminator operation for regions that appear in the body of
    `air.region` operation.  These regions are not expected to return any
    value so the terminator takes no operands.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

#endif
