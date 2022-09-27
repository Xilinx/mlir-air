//===- Runner.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
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

#include "air-c/Runner.h"

#include "air/Util/Runner.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

void airRunnerRun(MlirModule module, const char *jsonFileName, const char *outputFileName,
                  const char *topLevelFunction, bool verbose) {
  auto moduleOp = unwrap(module);
  std::string errorMessage;
  auto json_file = mlir::openInputFile(jsonFileName, &errorMessage);
  if (!json_file) {
    llvm::errs() << errorMessage << "\n";
    return;
  }

  auto output = mlir::openOutputFile(outputFileName, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return;
  }

  std::string json_str = json_file->getBuffer().str();
  llvm::StringRef sr(json_str);
  auto jsonModel = llvm::json::parse(sr);
  if (!jsonModel) {
    llvm::errs() << "failed to parse model json\n";
    return;
  }

  xilinx::air::AIRRunner runner(output->os(), *jsonModel, verbose);

  auto toplevel = moduleOp.lookupSymbol<mlir::func::FuncOp>(topLevelFunction);
  if (!toplevel) {
    llvm::errs() << "Function not supported.\n";
    return;
  }

  runner.emitTraceStart(output->os());
  runner.scheduleFunction(toplevel);
  runner.emitTraceEnd(output->os());

  output->keep();
  return;
}
