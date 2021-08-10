// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#ifndef AFFINE_TO_AIR_H_
#define AFFINE_TO_AIR_H_

#include <memory>

namespace mlir {
class Pass;
class DialectConversion;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAffineToAIRPass();
  void registerAffineToAIRPass();
} // namespace air
} // namespace xilinx

#endif // AFFINE_TO_AIR_H_
