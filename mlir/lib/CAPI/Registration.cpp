// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "air-c/Registration.h"

#include "air/InitAll.h"

#include "mlir/CAPI/IR.h"
#include "mlir/InitAllPasses.h"

void airRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  xilinx::air::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  // TODO: Don't eagerly load once D88162 is in and clients can do this.
  unwrap(context)->loadAllAvailableDialects();
}

void airRegisterAllPasses() {
  xilinx::air::registerAllPasses();
}
