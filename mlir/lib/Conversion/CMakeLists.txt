
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
AIRDialect
AIRRtDialect
AIRUtil
AIE
MLIRAffine
MLIRIR
MLIRLinalg
MLIRLinalgUtils
MLIRLinalgTransforms
MLIRLLVMIR
MLIRSCF
MLIRSupport
MLIRTransforms
)