//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_OPS
#define AIRRT_OPS

include "AIRRtBase.td"
include "mlir/IR/SymbolInterfaces.td"

class AIRRt_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<AIRRt_Dialect, mnemonic, traits> {
}

def AIRRt_ModuleMetadataOp : AIRRt_Op<"module_metadata", [
  SingleBlockImplicitTerminator<"ModuleMetadataTerminatorOp">
]> {
  let summary = "Global metadata for the module";
  let description = [{
    This op contains a region containing airrt.herd_metadata ops,
    which give information about the herds in the module.

    There is one of these per module.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$herds);

  let printer = [{ return ::print$cppClass(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def AIRRt_ModuleMetadataTerminatorOp
    : AIRRt_Op<"module_metadata_terminator",
               [Terminator, HasParent<"ModuleMetadataOp">]> {
  let summary = "Implicit terminator for ModuleMetadataOp's region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def AIRRt_HerdMetadataOp
    : AIRRt_Op<"herd_metadata", [
        HasParent<"ModuleMetadataOp">,
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
    let arguments = (ins StrAttr:$sym_name, Variadic<AnyType>:$args);
    let results = (outs I32:$ret);
    let assemblyFormat = [{
      $sym_name `(` $args `)` attr-dict `:` functional-type($args, results)

    }];
}

def AIRRt_DmaMemcpyOp : AIRRt_Op<"dma_memcpy", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32:$id,
        I64:$x,
        I64:$y,
        AnyMemRef:$memref,
        I64:$offset,
        I64:$num
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$memref
    `[` $offset `]` `,`
    $num `)`  attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($memref)`,`
    `[` type($offset) `]` `,`
    type($num) `)`  `->` `(` `)`
  }];
  let description = [{
    half dma operator
  }];
}

def AIRRt_DmaMemcpy2dOp: AIRRt_Op<"dma_memcpy_2d", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32:$id,
        I64:$x,
        I64:$y,
        AnyMemRef:$memref,
        I64:$offset1,
        I64:$offset0,
        I64:$num,
        I64:$stride,
        I64:$elem_per_stride
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$memref
    `[` $offset1`,`$offset0 `]` `,`
    $num `,` $stride `,` $elem_per_stride `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($memref)`,`
    `[` type($offset1)`,` type($offset0) `]` `,`
    type($num) `,` type($stride) `,` type($elem_per_stride) `)`  `->` `(` `)`
  }];
  let description = [{
    2d half dma operator
  }];
}

def AIRRt_DmaMemcpy4dOp: AIRRt_Op<"dma_memcpy_4d", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32:$id,
        I64:$x,
        I64:$y,
        AnyMemRef:$memref,
        I64:$offset3,
        I64:$offset2,
        I64:$offset1,
        I64:$offset0,
        I64:$num,
        I64:$stride,
        I64:$elem_per_stride
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$memref
    `[` $offset3`,`$offset2`,`$offset1`,`$offset0 `]` `,`
    $num `,` $stride `,` $elem_per_stride `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($memref)`,`
    `[` type($offset3)`,`type($offset2)`,`type($offset1)`,` type($offset0) `]` `,`
    type($num) `,` type($stride) `,` type($elem_per_stride) `)`  `->` `(` `)`
  }];
  let description = [{
    4d half dma operator
  }];
}

#endif // #ifndef AIRRT_OPS
