#ifndef AIR_TRANSFORM_PASSES_H
#define AIR_TRANSFORM_PASSES_H

#include "air/Transform/AffineLoopOptPass.h"
#include "air/Transform/AIRAutomaticTilingPass.h"
#include "air/Transform/AIRHerdAssignPass.h"
#include "air/Transform/AIRLinalgCodegen.h"
#include "air/Transform/AIRLoopMergingPass.h"
#include "air/Transform/AIRLoopPermutationPass.h"
#include "air/Transform/AIRRegularizeLoopPass.h"
#include "air/Transform/AIRTilingUtils.h"
#include "air/Transform/ReturnEliminationPass.h"

namespace xilinx {
namespace air {

void registerTransformPasses();

} // namespace air
} // namespace xilinx

#endif // AIR_TRANSFORM_PASSES_H