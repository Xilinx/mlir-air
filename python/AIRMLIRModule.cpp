//===- AIRMLIRModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "air-c/Dialects.h"
#include "air-c/Registration.h"
#include "air-c/Runner.h"
#include "air-c/Transform.h"

#include "mlir-c/Diagnostics.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir::python;

NB_MODULE(_air, m) {

  ::airRegisterAllPasses();

  m.doc() = R"pbdoc(
    AIR MLIR Python bindings
    --------------------------

    .. currentmodule:: _air

    .. autosummary::
        :toctree: _generate
  )pbdoc";

  m.def(
      "register_dialect",
      [](MlirDialectRegistry registry) { airRegisterAllDialects(registry); },
      "registry"_a);

  // AIR types bindings
  nanobind_adaptors::mlir_type_subclass(m, "AsyncTokenType",
                                        mlirTypeIsAIRAsyncTokenType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext ctx) {
            return cls(mlirAIRAsyncTokenTypeGet(ctx));
          },
          "Get an instance of AsyncTokenType in given context.",
          nb::arg("self"), nb::arg("ctx") = nb::none());

  // Raises if the transform fails, with the diagnostics in the message.
  m.def(
      "run_transform",
      [](MlirModule transform, MlirModule payload) {
        MlirContext ctx = mlirModuleGetContext(payload);
        std::string diags;
        auto handler = mlirContextAttachDiagnosticHandler(
            ctx,
            [](MlirDiagnostic diag, void *userData) {
              auto append = [](MlirStringRef str, void *data) {
                static_cast<std::string *>(data)->append(str.data, str.length);
              };
              auto &out = *static_cast<std::string *>(userData);
              mlirLocationPrint(mlirDiagnosticGetLocation(diag), append,
                                userData);
              out += ": ";
              mlirDiagnosticPrint(diag, append, userData);
              out += "\n";
              return mlirLogicalResultSuccess();
            },
            &diags, nullptr);
        MlirLogicalResult result = ::runTransform(transform, payload);
        mlirContextDetachDiagnosticHandler(ctx, handler);
        if (mlirLogicalResultIsFailure(result))
          throw std::runtime_error("transform failed:\n" + diags);
        // Warnings and remarks from a successful run still go to stderr.
        std::fputs(diags.c_str(), stderr);
      },
      "transform"_a, "payload"_a);

  m.attr("__version__") = "dev";

  // AIR Runner bindings
  auto air_runner = m.def_submodule("runner", "air-runner bindings");
  air_runner.def("run", [](MlirModule module, const std::string &json,
                           const std::string &outfile,
                           const std::string &function,
                           const std::string &sim_granularity,
                           const std::string &launch_iterations, bool verbose) {
    airRunnerRun(module, json.c_str(), outfile.c_str(), function.c_str(),
                 sim_granularity.c_str(), launch_iterations.c_str(), verbose);
  });
}
