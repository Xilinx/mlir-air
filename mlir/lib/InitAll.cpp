// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "air/InitAll.h"

#include "air/Conversion/Passes.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Transform/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"

void xilinx::air::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<xilinx::air::airDialect, xilinx::airrt::AIRRtDialect>();
}

void xilinx::air::registerAllPasses() {
  xilinx::air::registerTransformPasses();
  xilinx::air::registerConversionPasses();
}
