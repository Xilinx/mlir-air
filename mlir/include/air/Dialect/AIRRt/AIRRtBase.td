//===- AIRRtBase.td ----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_BASE
#define AIRRT_BASE

include "mlir/IR/OpBase.td"

def AIRRt_Dialect : Dialect {
  let name = "airrt";
  let cppNamespace = "xilinx::airrt";
  let description = [{
The `airrt` dialect is an IR for interacting with the AIR runtime.
It enapsulates data structures and functions available in the runtime and
can be lowered to a combination of standard and LLVM dialects.
  }];
  let useDefaultTypePrinterParser = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

def AIRRt_Event : DialectType<
  AIRRt_Dialect, CPred<"$_self.isa<xilinx::airrt::EventType>()">, "event type">,
             BuildableType<"xilinx::airrt::EventType::get($_builder.getContext())">;

#endif // #ifndef AIRRT_BASE
