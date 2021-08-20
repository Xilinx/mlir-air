// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "air/Transform/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "air/Transform/Passes.h.inc"
}

void xilinx::air::registerTransformPasses() { ::registerAIRTransformPasses(); }
