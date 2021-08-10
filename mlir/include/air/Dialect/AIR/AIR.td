// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.


#ifndef AIR_OPS
#define AIR_OPS

include "mlir/IR/OpBase.td"
include "AIROpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "npcomp/Dialect/ATen/IR/ATenOpInterface.td"

class air_Op<string mnemonic, list<OpTrait> traits = [StatisticsOpInterface]> :
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
    Defines and runs a 2D grid of tiles as an AIR Herd.
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

def air_AllocOp: air_Op<"alloc", []>,
                Results<(outs AnyMemRef)> {
  let arguments = (
    ins Variadic<AnyType>:$events
  );

  let summary = "alloc operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

def air_DeallocOp: air_Op<"dealloc", []>,
                Results<(outs I1:$evt)> {
  let arguments = (
    ins AnyMemRef:$memref,
        Variadic<AnyType>:$events
  );

  let summary = "dealloc operator";
  let description = [{
    dma operator
  }];
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


def air_DmaCopyOp: air_Op<"copy", []>,
                Results<(outs I1:$evt)> {
  let arguments = (
    ins AnyMemRef:$dst,
        AnyMemRef:$src,
        Variadic<AnyType>:$events
  );

  let summary = "dma operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

def air_WaitAllOp: air_Op<"wait_all", []>,
                Results<(outs Optional<air_AsyncToken>:$asyncToken)> {
  let arguments = (ins Variadic<air_AsyncToken>:$asyncDependencies);

  let summary = "wait for all operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
  }];
}

def air_DmaLoadOp: air_Op<"tensor_load", []>,
                Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyMemRef:$memref,
        Variadic<AnyType>:$events
  );

  let summary = "dma operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

def air_DmaStoreOp: air_Op<"tensor_store", []>,
                Results<(outs I1:$evt)> {
  let arguments = (
    ins AnyMemRef:$memref,
        AnyTensor:$tensor,
        Variadic<AnyType>:$events
  );

  let summary = "dma operator";
  let description = [{
    dma operator
  }];
//   let extraClassDeclaration = [{
//     std::map<std::string, uint64_t> getStatistics();
//   }];
}

#endif
