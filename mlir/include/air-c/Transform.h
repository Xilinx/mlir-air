//===- Transform.h ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_C_TRANSFORM_H
#define AIR_C_TRANSFORM_H

#include "mlir-c/IR.h"

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void runTransform(MlirModule transform, MlirModule payload);

#ifdef __cplusplus
}
#endif

#endif // AIR_C_TRANSFORM_H
