// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "air-c/Registration.h"

#include "mlir/CAPI/IR.h"
#include "mlir/InitAllPasses.h"
#include "air/InitAll.h"

void airRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  xilinx::air::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  // TODO: Don't eagerly load once D88162 is in and clients can do this.
  unwrap(context)->loadAllAvailableDialects();
}

void airRegisterAllPasses() {
  xilinx::air::registerAllPasses();
  //mlir::registerAllPasses();
}
