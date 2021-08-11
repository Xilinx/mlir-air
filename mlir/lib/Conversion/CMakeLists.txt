
add_air_library(AIRConversionPasses
AffineToAIRPass.cpp
AIRLoweringPass.cpp
AIRRtToLLVMPass.cpp
AIRToAIEPass.cpp
ATenToXTenPass.cpp
XTenToAffine.cpp
XTenToLinalg.cpp

DEPENDS
AIRConversionPassIncGen
AIRATenToXTenIncGen


LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
${air_conversion_libs}
)