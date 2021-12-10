// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "air-c/Dialects.h"
#include "air-c/Registration.h"

namespace py = pybind11;

PYBIND11_MODULE(_airMlir, m) {

  ::airRegisterAllPasses();

  m.doc() = R"pbdoc(
    Xilinx AIR MLIR Python bindings
    --------------------------

    .. currentmodule:: AIRMLIR_

    .. autosummary::
        :toctree: _generate
  )pbdoc";
  
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__air__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def("_register_all_passes", ::airRegisterAllPasses);
  m.attr("__version__") = "dev";
}
