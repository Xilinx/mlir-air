// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.
#ifndef AIR_C_RUNNER_H
#define AIR_C_RUNNER_H

#include "mlir-c/IR.h"

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

void airRunnerRun(MlirModule module, const char *json_file_name,
                  const char *output_file_name, const char *function, bool verbose);

#ifdef __cplusplus
}
#endif

#endif // AIR_C_RUNNER_H
