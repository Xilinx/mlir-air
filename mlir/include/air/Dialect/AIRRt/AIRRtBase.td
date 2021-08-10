// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#ifndef AIRRT_BASE
#define AIRRT_BASE

include "mlir/IR/OpBase.td"

def AIRRt_Dialect : Dialect {
  let name = "airrt";
  let cppNamespace = "xilinx::airrt";
  let description = [{
The `airrt` dialect is an IR for interacting with the AIR runtime.
It enapsulates data structures and functions available in the runtime and
can be lowered to LLVM dialect.
  }];
}

#endif // #ifndef AIRRT_BASE
