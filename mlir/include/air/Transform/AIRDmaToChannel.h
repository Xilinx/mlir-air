//===- AIRDmaToChannel.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DMA_TO_CHANNEL_H
#define AIR_DMA_TO_CHANNEL_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createDmaToChannelPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DMA_TO_CHANNEL_H