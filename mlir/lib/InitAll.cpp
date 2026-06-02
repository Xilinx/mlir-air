//===- InitAll.cpp ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/InitAll.h"

#include "air/Conversion/Passes.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Interfaces/AIRInterfaces.h"
#include "air/Transform/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"

#if AIR_ENABLE_AIE
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#endif

void xilinx::air::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<xilinx::air::airDialect, xilinx::airrt::AIRRtDialect>();
  xilinx::air::registerTransformDialectExtension(registry);
  xilinx::air::registerCodegenInterfaces(registry);
}

void xilinx::air::registerAllPasses() {
  xilinx::air::registerTransformPasses();
  xilinx::air::registerConversionPasses();
#if AIR_ENABLE_AIE
  // Register mlir-aie's transform passes (most importantly aie-place-tiles)
  // so air-opt and aircc can invoke them. AIR emits aie.logical_tile<...>
  // for memtiles and shim DMA tiles; aie-place-tiles resolves these to
  // physical aie.tile ops.
  xilinx::AIE::registerAIEPasses();
#endif
}
