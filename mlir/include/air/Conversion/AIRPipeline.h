//===- AIRPipeline.h --------------------------------------------*- C++ -*-===//
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