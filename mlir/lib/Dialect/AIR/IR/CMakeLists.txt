add_air_dialect_library(AIRDialect
  AIRDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/mlir/include/air/Dialect/AIR

  DEPENDS
  MLIRAIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
)