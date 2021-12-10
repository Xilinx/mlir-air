// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "air-c/Dialects.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIR, air, xilinx::air::airDialect)
