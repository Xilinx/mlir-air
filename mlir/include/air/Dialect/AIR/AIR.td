// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.


#ifndef AIR_OPS
#define AIR_OPS

include "mlir/IR/OpBase.td"
include "AIROpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

class air_Op<string mnemonic, list<OpTrait> traits = []> :
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
    OpBuilder<(ins "HerdDim2":$gridSize,"ValueRange":$kernelOperands)>
  ];
  let parser = [{ return parseHerdLaunchOp(parser, result); }];
  let printer = [{ printHerdLaunchOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let extraClassDeclaration = [{
    HerdDim2 getTileIds();
    HerdDim2 getHerdSize();
    HerdDim2 getHerdSizeOperands();
    unsigned getNumKernelOperands();
    Value getKernelOperand(unsigned i);
    ArrayRef<BlockArgument> getKernelArguments();
  }];
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

  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
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

  let assemblyFormat = [{
    attr-dict $body
  }];
}

def air_PipelineStageOp : air_Op<"pipeline.stage", [HasParent<"HerdPipelineOp">]>,
    Arguments<(ins Variadic<AnyType>:$opers)>, Results<(outs Variadic<AnyType>:$results)> {
  let summary = "Pipeline stage";
  let regions = (region AnyRegion:$body);
  // let assemblyFormat = [{
  //   `(` $opers `)` `:` type($opers) attr-dict $body `->` `(` type($results) `)`
  // }];
  let description = [{
    Pipeline stage.
  }];
  let parser = [{ return parsePipelineStageOp(parser, result); }];
  let printer = [{ printPipelineStageOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
}

def air_PipelineYieldOp : air_Op<"pipeline.yield", [HasParent<"PipelineStageOp">,
                                          NoSideEffect, Terminator]>,
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
    Arguments<(ins AnyType:$dst, Variadic<AnyType>:$opers)>, Results<(outs)> {
  let summary = "Put for air pipeline stages.";
  let description = [{
    Experimental operation to represent copying data to another tile.
    Currently used internally by air-to-aie pass during pipeline lowering.
  }];
  let assemblyFormat = [{
    $dst (`,` $opers^)? attr-dict `:` type($dst) (`,` type($opers)^)?
  }];
}

def air_PipelineGetOp : air_Op<"pipeline.get", []>,
    Arguments<(ins AnyType:$src)>, Results<(outs Variadic<AnyType>:$results)> {
  let summary = "Get for air pipeline stages.";
  let description = [{
    Experimental operation to represent copying data from another tile.
    Currently used internally by air-to-aie pass during pipeline lowering.
  }];
  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($results)
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

  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
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
    Value getLength() { return num(); }
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
    Value getLength() { return num(); }
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
    Value getLength() { return num(); }
    int32_t getId() {
      if (auto id_attr = (*this)->getAttrOfType<IntegerAttr>("id")) {
        return id_attr.getInt();
      }
      return -1;
    }
  }];
}

def air_WaitAllOp: air_Op<"wait_all", []>,
                Results<(outs Optional<air_AsyncToken>:$asyncToken)> {
  let arguments = (ins Variadic<air_AsyncToken>:$asyncDependencies);

  let summary = "wait for all operator";
  let description = [{
    Block and wait for all async tokens before preceding.
  }];
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
  }];
}

#endif
