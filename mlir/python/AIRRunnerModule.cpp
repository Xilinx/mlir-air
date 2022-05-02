
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <iostream>

#include "AIRRunnerModule.h"

#include "air-c/Runner.h"

namespace xilinx {
namespace air {

void defineAIRRunnerModule(pybind11::module &m) { m.def("run", airRunnerRun); }

} // namespace air
} // namespace xilinx