//===- Module.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include "LibAirHostModule.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace {

} // namespace

PYBIND11_MODULE(_airRt, m) {
  m.doc() = R"pbdoc(
        AIR Runtime Python bindings
        --------------------------

        .. currentmodule:: _airRt

        .. autosummary::
           :toctree: _generate

    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

  auto airhost = m.def_submodule("host", "libairhost bindings");
  xilinx::air::defineAIRHostModule(airhost);
}
