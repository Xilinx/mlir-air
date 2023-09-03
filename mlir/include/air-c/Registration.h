//===- Registration.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_C_REGISTRATION_H
#define AIR_C_REGISTRATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all AIR dialects with a context.
 * This is needed before creating IR for these Dialects.
 */
MLIR_CAPI_EXPORTED void airRegisterAllDialects(MlirContext context);

/** Registers all AIR passes for symbolic access with the global registry. */
MLIR_CAPI_EXPORTED void airRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // AIR_C_REGISTRATION_H
