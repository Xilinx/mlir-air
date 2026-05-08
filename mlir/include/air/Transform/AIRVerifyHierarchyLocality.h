//===- AIRVerifyHierarchyLocality.h -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_VERIFY_HIERARCHY_LOCALITY_H
#define AIR_VERIFY_HIERARCHY_LOCALITY_H

#include "air/Transform/PassDetail.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRVerifyHierarchyLocalityPass();

} // namespace air
} // namespace xilinx

#endif // AIR_VERIFY_HIERARCHY_LOCALITY_H
