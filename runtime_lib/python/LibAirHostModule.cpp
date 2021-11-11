

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <iostream>

#include "LibAirHostModule.h"

#ifdef AIE_LIBXAIE_ENABLE
#include "air_host.h"
#endif

namespace xilinx {
namespace air {

void defineAIRHostModule(pybind11::module &m) {
#ifdef AIE_LIBXAIE_ENABLE
  m.def("init_libxaie1", [](void) -> void {
    air_init_libxaie1();
  });
  m.def("deinit_libxaie1", &air_deinit_libxaie1);

  pybind11::class_<air_module_desc_t>(m, "ModuleDescriptor")
    .def("getHerds", [](const air_module_desc_t &d) -> std::vector<air_herd_desc_t*> {
      std::vector<air_herd_desc_t*> herds;
      for (uint64_t i=0; i<d.length; i++)
        herds.push_back(d.herd_descs[i]);
      return herds;
    }, pybind11::return_value_policy::reference);

  pybind11::class_<air_herd_desc_t>(m, "HerdDescriptor")
    .def("getName", [](const air_herd_desc_t &d) -> std::string {
      return std::string(d.name, d.name_length);
    });

  m.def("module_load_from_file", [](std::string filename) -> air_module_handle_t {
    return air_module_load_from_file(filename.c_str(),nullptr);
  });

  m.def("module_unload", &air_module_unload);

  m.def("get_module_descriptor", &air_module_get_desc,
        pybind11::return_value_policy::reference);

  m.def("get_herd_descriptor", [](air_module_handle_t h, std::string name) {
    return air_herd_get_desc(h, name.c_str());
  }, pybind11::return_value_policy::reference);
#endif
}

}
}
