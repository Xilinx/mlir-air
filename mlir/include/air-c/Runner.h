// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.
#ifndef AIR_C_RUNNER_H
#define AIR_C_RUNNER_H

#include "mlir-c/IR.h"

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

void airRunnerRun(MlirModule module, char *json_file_name,
                  char *output_file_name, char *function, bool verbose);

#ifdef __cplusplus
}
#endif

#endif // AIR_C_RUNNER_H
