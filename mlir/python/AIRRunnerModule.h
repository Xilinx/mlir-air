#ifndef AIR_RUNNER_MODULE_H
#define AIR_RUNNER_MODULE_H

#include <pybind11/pybind11.h>

namespace xilinx {
namespace air {

void defineAIRRunnerModule(pybind11::module &m);

}
} // namespace xilinx

#endif // AIR_RUNNER_MODULE_H