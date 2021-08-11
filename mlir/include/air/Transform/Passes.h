#ifndef AIR_TRANSFORM_PASSES_H
#define AIR_TRANSFORM_PASSES_H

#include "air/Transform/AffineLoopOptPass.h"
#include "air/Transform/AIRAutomaticTilingPass.h"
#include "air/Transform/AIRLinalgCodegen.h"
#include "air/Transform/AIRLoopPermutationPass.h"
#include "air/Transform/AIRRegularizeLoopPass.h"

namespace xilinx {
namespace air {

using namespace mlir;

#define GEN_PASS_REGISTRATION
#include "air/Transform/Passes.h.inc"

} // namespace air
} // namespace xilinx

#endif // AIR_TRANSFORM_PASSES_H