//===- LibAirHostModule.cpp -------------------------------------*- C++ -*-===//
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
