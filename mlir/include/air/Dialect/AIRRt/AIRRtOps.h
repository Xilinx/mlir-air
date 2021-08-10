// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#ifndef AIRRTOPS_H
#define AIRRTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "air/Dialect/AIRRt/AIRRtOps.h.inc"

#endif // AIRRTOPS_H
