//===- AIRMLIRModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "air-c/Dialects.h"
#include "air-c/Registration.h"

#include "AIRRunnerModule.h"

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
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__air__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def("_register_all_passes", ::airRegisterAllPasses);

  // AIR types bindings
  mlir_type_subclass(m, "AsyncTokenType", mlirTypeIsAIRAsyncTokenType)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx) {
            return cls(mlirAIRAsyncTokenTypeGet(ctx));
          },
          "Get an instance of AsyncTokenType in given context.",
          py::arg("self"), py::arg("ctx") = py::none());

  m.attr("__version__") = "dev";

  // AIR Runner bindings
  auto air_runner = m.def_submodule("runner", "air-runner bindings");
  xilinx::air::defineAIRRunnerModule(air_runner);
}
