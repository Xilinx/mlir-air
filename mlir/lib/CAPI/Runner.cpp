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
