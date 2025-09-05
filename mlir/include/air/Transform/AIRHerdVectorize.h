//===- AIRHerdVectorize.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_HERD_VECTORIZE_H
#define AIR_HERD_VECTORIZE_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHerdVectorizePass();
std::unique_ptr<mlir::Pass>
createAIRHerdVectorizePass(bool vectorizeNdExtract, bool flatten1dDepthwiseConv,
                           bool disableTransferPermutationMapLoweringPatterns,
                           bool disableMultiReductionToContractPatterns,
                           bool vectorizePadding);

} // namespace air
} // namespace xilinx

#endif // AIR_HERD_VECTORIZE_H
