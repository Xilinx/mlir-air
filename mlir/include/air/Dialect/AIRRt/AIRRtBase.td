//===- AIRRtBase.td ----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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
}

def AIRRt_Event : DialectType<
  AIRRt_Dialect, CPred<"$_self.isa<xilinx::airrt::EventType>()">, "event type">,
             BuildableType<"xilinx::airrt::EventType::get($_builder.getContext())">;

#endif // #ifndef AIRRT_BASE
