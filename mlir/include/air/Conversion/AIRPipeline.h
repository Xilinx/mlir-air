//===- AIRPipeline.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_PIPELINE_H
#define AIR_PIPELINE_H

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace xilinx {
namespace air {

class AIRPipeStageConversion : public ConversionPattern {
public:
  enum LoweringType { AllocBuffer = 0, PipelineGetPut = 1 };

  explicit AIRPipeStageConversion(MLIRContext *context, LoweringType type)
      : ConversionPattern(xilinx::air::PipelineStageOp::getOperationName(), 10,
                          context),
        loweringType(type) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

private:
  LoweringType loweringType;
};

} // namespace air
} // namespace xilinx
#endif // AIR_PIPELINE_H