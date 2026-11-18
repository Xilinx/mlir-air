//===- AIRRtDialect.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIRRT_DIALECT_H
#define AIRRT_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "air/Dialect/AIRRt/AIRRtOpsTypes.h.inc"

#include "air/Dialect/AIRRt/AIRRtOpsDialect.h.inc"

#endif // AIRRT_DIALECT_H
