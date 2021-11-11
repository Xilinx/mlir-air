declare_mlir_python_sources(AirCompilerPythonSources)

declare_mlir_python_sources(AirCompilerPythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT AirCompilerPythonSources
  SOURCES_GLOB
    aircc/*.py
    *.py
)

add_mlir_python_modules(AirCompilerPythonModules
  ROOT_PREFIX "${AIR_PYTHON_PACKAGES_DIR}/air/compiler"
  INSTALL_PREFIX "python_packages/air/compiler"
  DECLARED_SOURCES AirCompilerPythonSources
  )
