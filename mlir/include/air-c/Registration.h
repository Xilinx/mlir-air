// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_C_REGISTRATION_H
#define AIR_C_REGISTRATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all AIR dialects with a context.
 * This is needed before creating IR for these Dialects.
 */
void airRegisterAllDialects(MlirContext context);

/** Registers all AIR passes for symbolic access with the global registry. */
void airRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // AIR_C_REGISTRATION_H
