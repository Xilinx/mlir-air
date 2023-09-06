//===- AIRMLIRModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "air-c/Dialects.h"
#include "air-c/Registration.h"
#include "air-c/Runner.h"
#include "air-c/Transform.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_airMlir, m) {

  ::airRegisterAllPasses();

  m.doc() = R"pbdoc(
    AIR MLIR Python bindings
    --------------------------

    .. currentmodule:: _airMlir

    .. autosummary::
        :toctree: _generate
  )pbdoc";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) { airRegisterAllDialects(context); },
      py::arg("context"), py::arg("load") = true);

  m.def("_register_all_passes", ::airRegisterAllPasses);

  // AIR types bindings
  mlir_type_subclass(m, "AsyncTokenType", mlirTypeIsAIRAsyncTokenType)
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirContext ctx) {
            return cls(mlirAIRAsyncTokenTypeGet(ctx));
          },
          "Get an instance of AsyncTokenType in given context.",
          py::arg("self"), py::arg("ctx") = py::none());

  m.def("_run_air_transform", ::runTransform);

  m.attr("__version__") = "dev";

  // AIR Runner bindings
  auto air_runner = m.def_submodule("runner", "air-runner bindings");
  air_runner.def("run", [](MlirModule module, const std::string &json,
                           const std::string &outfile,
                           const std::string &function,
                           const std::string &sim_granularity, bool verbose) {
    airRunnerRun(module, json.c_str(), outfile.c_str(), function.c_str(),
                 sim_granularity.c_str(), verbose);
  });
}
