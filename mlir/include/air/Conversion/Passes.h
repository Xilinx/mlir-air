#ifndef AIR_CONVERSION_PASSES_H
#define AIR_CONVERSION_PASSES_H

#include "air/Conversion/AffineToAIRPass.h"
#include "air/Conversion/AIRLoweringPass.h"
#include "air/Conversion/AIRRtToLLVMPass.h"
#include "air/Conversion/XTenToAffinePass.h"
#include "air/Conversion/XTenToLinalgPass.h"

namespace xilinx {
namespace air {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "air/Conversion/Passes.h.inc"

} // namespace air
} // namespace xilinx

#endif // AIR_CONVERSION_PASSES_H