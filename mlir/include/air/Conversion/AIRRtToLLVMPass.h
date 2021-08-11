// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIRRT_TO_LLVM_H
#define AIRRT_TO_LLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace airrt {

std::unique_ptr<mlir::Pass> createAIRRtToLLVMPass();

} // namespace airrt
} // namespace xilinx

#endif // AIRRT_TO_LLVM_H