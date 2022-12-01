//===- AIRTargets.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace xilinx {
namespace air {

mlir::LogicalResult AIRHerdsToJSON(mlir::ModuleOp module,
                                   llvm::raw_ostream &output);

}
} // namespace xilinx