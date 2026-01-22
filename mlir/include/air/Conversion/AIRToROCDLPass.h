//===- AIRToROCDLPass.h - Convert AIR to ROCDL Pass -----------------*- C++ -*-===//
//
// This file declares the pass that converts AIR dialect to ROCDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERT_TO_AIR_ROCDL
#define CONVERT_TO_AIR_ROCDL

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToROCDLPass();

} // namespace air
} // namespace xilinx
#endif // CONVERT_TO_AIR_ROCDL
