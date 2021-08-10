add_air_dialect_library(AIRRtDialect
  AIRRtDialect.cpp
  AIRRtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/AIRRt

  DEPENDS
  MLIRAIRRtOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
)