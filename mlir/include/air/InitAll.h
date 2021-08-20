// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_INITALL_H
#define AIR_INITALL_H

#include "mlir/IR/Dialect.h"

namespace xilinx {
namespace air {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace air
} // namespace xilinx

#endif // AIR_INITALL_H
