
#include "AIRRunnerModule.h"

#include "air-c/Runner.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <string>

namespace xilinx {
namespace air {

void defineAIRRunnerModule(pybind11::module &m) {
  m.def("run", [](MlirModule module, std::string json, std::string outfile,
                  std::string function, bool verbose) {
    airRunnerRun(module, json.c_str(), outfile.c_str(), function.c_str(),
                 verbose);
  });
}

} // namespace air
} // namespace xilinx