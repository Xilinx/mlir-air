// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <iostream>

#include "LibAirHostModule.h"

#ifdef AIE_LIBXAIE_ENABLE
#include "air_host.h"
#include "acdc_queue.h"
#endif
namespace xilinx {
namespace air {

void defineAIRHostModule(pybind11::module &m) {
#ifdef AIE_LIBXAIE_ENABLE

  pybind11::class_<aie_libxaie_ctx_t>(m, "LibXAIEContext");

  m.def("init_libxaie", &air_init_libxaie1, pybind11::return_value_policy::reference);

  m.def("deinit_libxaie",[](aie_libxaie_ctx_t* ctx) -> void {
    air_deinit_libxaie1(ctx);
  });

  pybind11::class_<air_module_desc_t>(m, "ModuleDescriptor")
      .def(
          "getPartitions",
          [](const air_module_desc_t &d)
              -> std::vector<air_partition_desc_t *> {
            std::vector<air_partition_desc_t *> partitions;
            for (uint64_t i = 0; i < d.partition_length; i++)
              partitions.push_back(d.partition_descs[i]);
            return partitions;
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<air_partition_desc_t>(m, "PartitionDescriptor")
      .def(
          "getHerds",
          [](const air_partition_desc_t &d) -> std::vector<air_herd_desc_t *> {
            std::vector<air_herd_desc_t *> herds;
            for (uint64_t i = 0; i < d.herd_length; i++)
              herds.push_back(d.herd_descs[i]);
            return herds;
          },
          pybind11::return_value_policy::reference)
      .def("getName", [](const air_partition_desc_t &d) -> std::string {
        return std::string(d.name, d.name_length);
      });

  pybind11::class_<air_herd_desc_t>(m, "HerdDescriptor")
    .def("getName", [](const air_herd_desc_t &d) -> std::string {
      return std::string(d.name, d.name_length);
    });

  m.def("module_load_from_file", [](std::string filename, queue_t* q) -> air_module_handle_t {
    return air_module_load_from_file(filename.c_str(), q);
  });

  m.def("module_unload", &air_module_unload);

  m.def("get_module_descriptor", &air_module_get_desc,
        pybind11::return_value_policy::reference);

  // m.def("get_herd_descriptor", [](air_module_handle_t h, std::string name) {
  //   return air_herd_get_desc(h, name.c_str());
  // }, pybind11::return_value_policy::reference);

  pybind11::class_<queue_t>(m, "Queue");

  m.def("queue_create", []() -> queue_t* {
    queue_t *q = nullptr;
    auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
    if (ret != 0)
      return nullptr;
    return q;
  }, pybind11::return_value_policy::reference);
#endif
}

}
}
