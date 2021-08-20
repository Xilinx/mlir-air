// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "air/Conversion/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "air/Conversion/Passes.h.inc"
}

void xilinx::air::registerConversionPasses() { ::registerAIRConversionPasses(); }
