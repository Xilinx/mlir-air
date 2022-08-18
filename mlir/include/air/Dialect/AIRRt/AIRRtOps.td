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

class AIRRt_Op<string mnemonic, list<Trait> traits = []>
    : Op<AIRRt_Dialect, mnemonic, traits> {
}

def AIRRt_ModuleMetadataOp : AIRRt_Op<"module_metadata", [
  SingleBlockImplicitTerminator<"ModuleMetadataTerminatorOp">
]> {
  let summary = "Global metadata for the module";
  let description = [{
    This op contains a region containing airrt.partition_metadata ops,
    which give information about the partitions in the module.

    There is one of these per module.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$partitions);
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

def AIRRt_PartitionMetadataOp : AIRRt_Op<"partition_metadata", [
  HasParent<"ModuleMetadataOp">,
  SingleBlockImplicitTerminator<"PartitionMetadataTerminatorOp">
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

def AIRRt_PartitionMetadataTerminatorOp
    : AIRRt_Op<"partition_metadata_terminator",
               [Terminator, HasParent<"PartitionMetadataOp">]> {
  let summary = "Implicit terminator for PartitionMetadataOp's region";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def AIRRt_HerdMetadataOp
    : AIRRt_Op<"herd_metadata", [
        HasParent<"PartitionMetadataOp">,
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
    let arguments = (ins StrAttr:$sym_name);
    let results = (outs I64:$h, Optional<AIRRt_Event>:$event);
    let assemblyFormat = [{
      $sym_name attr-dict `:` type($h) (`,` type($event)^)?
    }];
}

def AIRRt_PartitionLoadOp : AIRRt_Op<"partition_load", []> {
    let summary = "load a partition";
    let arguments = (ins StrAttr:$sym_name);
    let results = (outs I64:$p, Optional<AIRRt_Event>:$event);
    let assemblyFormat = [{
      $sym_name attr-dict `:` type($p) (`,` type($event)^)?
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
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$memref
    `[` $offset `]` `,`
    $num `)`  attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($memref)`,`
    `[` type($offset) `]` `,`
    type($num) `)` (`:` type($event)^)?
  }];
  let description = [{
    half dma operator
  }];
}

def AIRRt_DmaMemcpyNdOp: AIRRt_Op<"dma_memcpy_nd", []> {
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
        I64:$length3,
        I64:$length2,
        I64:$length1,
        I64:$length0,
        I64:$stride3,
        I64:$stride2,
        I64:$stride1
  );
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$memref
    `[` $offset3`,`$offset2`,`$offset1`,`$offset0 `]` `,`
    `[` $length3`,`$length2`,`$length1`,`$length0 `]` `,`
    `[` $stride3`,`$stride2`,`$stride1 `]` `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($memref)`,`
    `[` type($offset3)`,`type($offset2)`,`type($offset1)`,` type($offset0) `]` `,`
    `[` type($length3)`,`type($length2)`,`type($length1)`,` type($length0) `]` `,`
    `[` type($stride3)`,`type($stride2)`,`type($stride1) `]` `)` (`:` type($event)^)?
  }];
  let description = [{
    nd half dma operator
  }];
}

def AIRRt_MemcpyNdOp: AIRRt_Op<"memcpy_nd", []> {
  let summary = "dma operator";
  let arguments = (
    ins AnyMemRef:$dst,
        AnyMemRef:$src,
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
        I64:$stride1
  );
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $dst`,`$src`,`
    `[` $offset3`,`$offset2`,`$offset1`,`$offset0 `]` `,`
    `[` $length3`,`$length2`,`$length1`,`$length0 `]` `,`
    `[` $stride3`,`$stride2`,`$stride1 `]` `)` attr-dict `:`
    `(` type($dst)`,`type($src)`,`
    `[` type($offset3)`,`type($offset2)`,`type($offset1)`,` type($offset0) `]` `,`
    `[` type($length3)`,`type($length2)`,`type($length1)`,` type($length0) `]` `,`
    `[` type($stride3)`,`type($stride2)`,`type($stride1) `]` `)`  (`:` type($event)^)?
  }];
  let description = [{
    nd memcpy operator
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
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$memref
    `[` $offset1`,`$offset0 `]` `,`
    $num `,` $stride `,` $elem_per_stride `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($memref)`,`
    `[` type($offset1)`,` type($offset0) `]` `,`
    type($num) `,` type($stride) `,` type($elem_per_stride) `)` (`:` type($event)^)?
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
  let results = (outs Optional<AIRRt_Event>:$event);
  let assemblyFormat = [{
    `(` $id `,`$x `,`$y `,`$memref
    `[` $offset3`,`$offset2`,`$offset1`,`$offset0 `]` `,`
    $num `,` $stride `,` $elem_per_stride `)` attr-dict `:`
    `(` type($id)`,`type($x)`,`type($y)`,`type($memref)`,`
    `[` type($offset3)`,`type($offset2)`,`type($offset1)`,` type($offset0) `]` `,`
    type($num) `,` type($stride) `,` type($elem_per_stride) `)` (`:` type($event)^)?
  }];
  let description = [{
    4d half dma operator
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
}

#endif // #ifndef AIRRT_OPS
