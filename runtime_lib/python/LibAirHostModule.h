#ifndef AIR_HOST_MODULE_H
#define AIR_HOST_MODULE_H

#include <pybind11/pybind11.h>

namespace xilinx {
namespace air {

void defineAIRHostModule(pybind11::module &m);

}
}

#endif // AIR_HOST_MODULE_H