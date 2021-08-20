
add_air_library(AIRTransformPasses
AffineLoopOptPass.cpp
AIRAutomaticTilingPass.cpp
AIRHerdAssignPass.cpp
AIRLinalgCodegen.cpp
AIRLoopMergingPass.cpp
AIRLoopPermutationPass.cpp
AIRTilingUtils.cpp
AIRRegularizeLoopPass.cpp
ReturnEliminationPass.cpp

DEPENDS
AIRTransformPassIncGen

LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
AIRUtil
${air_transform_libs}
)