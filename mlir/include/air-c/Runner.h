//===- Runner.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_C_RUNNER_H
#define AIR_C_RUNNER_H

#include "mlir-c/IR.h"

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void airRunnerRun(MlirModule module,
                                     const char *json_file_name,
                                     const char *output_file_name,
                                     const char *function,
                                     const char *sim_granularity, bool verbose);

#ifdef __cplusplus
}
#endif

#endif // AIR_C_RUNNER_H
