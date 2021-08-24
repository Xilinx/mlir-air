
add_air_library(AIRConversionPasses
AffineToAIRPass.cpp
AIRLoweringPass.cpp
AIRRtToLLVMPass.cpp
AIRToAIEPass.cpp
Passes.cpp

DEPENDS
AIRConversionPassIncGen
AIRDialect
AIRRtDialect

LINK_COMPONENTS
Core

LINK_LIBS
MLIR
AIE
)