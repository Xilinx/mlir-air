//===- Dialects.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_C_DIALECTS_H
#define AIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIR, air);

//===---------------------------------------------------------------------===//
// AsyncTokenType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAIRAsyncTokenType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirAIRAsyncTokenTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif // AIR_C_DIALECTS_H
