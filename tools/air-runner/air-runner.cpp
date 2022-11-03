//===- air-runner.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Runner.h"
#include "air/InitAll.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <vector>

#define DEBUG_TYPE "air-runner"

static bool verbose = false;

using namespace mlir;

namespace {

LogicalResult run(int argc, char **argv, llvm::StringRef toolName) {

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> jsonFileName(
      "m", llvm::cl::desc("json model filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init("arch.json"));

  static llvm::cl::opt<std::string> topLevelFunction(
      "f", llvm::cl::desc("top-level function name"),
      llvm::cl::value_desc("function"), llvm::cl::init("graph"));

  static llvm::cl::opt<bool> clVerbose("v", llvm::cl::desc("verbose"),
                                       llvm::cl::value_desc("bool"),
                                       llvm::cl::init(false));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);

  verbose = clVerbose;
  // herd_slots = clHerdSlots;
  // dispatch_slots = clDispatchSlots;

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto json_file = openInputFile(jsonFileName, &errorMessage);
  if (!json_file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           raw_ostream &os) {
    MLIRContext context;
    DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<xilinx::air::airDialect>();
    context.appendDialectRegistry(registry);

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());

    auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module)
      return failure();

    Operation *topOp = module->lookupSymbol(topLevelFunction);
    // The toplevel function can accept any number of operands, and returns
    // any number of results.
    if (!topOp) {
      llvm::errs() << "Toplevel function " << topLevelFunction
                   << " not found!\n";
      return failure();
    }

    // We need three things in a function-type independent way.
    // The type signature of the function.
    FunctionType ftype;
    // The arguments of the entry block.
    Block::BlockArgListType blockArgs;

    std::string json_str = json_file->getBuffer().str();
    StringRef sr(json_str);
    auto jsonModel = llvm::json::parse(sr);
    if (!jsonModel)
      llvm_unreachable("failed to parse model json\n");

    xilinx::air::AIRRunner runner(os, *jsonModel, clVerbose);

    // The number of inputs to the function in the IR.
    unsigned numInputs = 0;
    unsigned numOutputs = 0;

    if (func::FuncOp toplevel =
            module->lookupSymbol<func::FuncOp>(topLevelFunction)) {
      ftype = toplevel.getFunctionType();
      Block &entryBlock = toplevel.getBody().front();
      blockArgs = entryBlock.getArguments();

      // Get the primary inputs of toplevel off the command line.
      numInputs = ftype.getNumInputs();
      numOutputs = ftype.getNumResults();
    } else {
      llvm_unreachable("Function not supported.\n");
    }

    std::vector<std::string> inputArgs;

    runner.emitTraceStart(os);

    // for (unsigned i = 0; i < numInputs; i++) {
    //   Type type = ftype.getInput(i);
    //   if (auto tensorTy = type.dyn_cast<MemRefType>()) {
    //     // We require this memref type to be fully specified.
    //     std::vector<llvm::Any> nothing;
    //     unsigned buffer = runner.allocateMemRef(tensorTy, nothing);
    //     runner.valueMap[blockArgs[i]] = buffer;
    //     // runner.timeMap[blockArgs[i]] = 0;
    //   } else {
    //     llvm_unreachable("Only memref arguments are supported.\n");
    //   }
    // }

    std::vector<llvm::Any> results(numOutputs);
    std::vector<uint64_t> resultTimes(numOutputs);
    if (func::FuncOp toplevel =
            module->lookupSymbol<func::FuncOp>(topLevelFunction)) {
      runner.scheduleFunction(toplevel);
    }
    runner.emitTraceEnd(os);
    return success();
  };
  if (failed(processBuffer(std::move(input), output->os())))
    return failure();

  output->keep();
  return success();
}

} // namespace

int main(int argc, char *argv[]) {
  return failed(run(argc, argv, "AIR MLIR Modeling Tool"));
}