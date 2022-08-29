// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#ifndef AIR_C_DIALECTS_H
#define AIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIR, air);

#ifdef __cplusplus
}
#endif

#endif  // AIR_C_DIALECTS_H
