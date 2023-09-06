//===- AirHostModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air.hpp"

#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace {
void defineAIRHostModule(pybind11::module &m) {

  m.def(
      "init_libxaie", []() -> uint64_t { return (uint64_t)air_init_libxaie(); },
      pybind11::return_value_policy::reference);

  m.def("deinit_libxaie", [](uint64_t ctx) -> void {
    air_deinit_libxaie((air_libxaie_ctx_t)ctx);
  });

  m.def("init", []() -> uint64_t { return (uint64_t)air_init(); });

  m.def("shut_down", []() -> uint64_t { return (uint64_t)air_shut_down(); });

  pybind11::class_<air_module_desc_t>(m, "ModuleDescriptor")
      .def(
          "getSegments",
          [](const air_module_desc_t &d) -> std::vector<air_segment_desc_t *> {
            std::vector<air_segment_desc_t *> segments;
            for (uint64_t i = 0; i < d.segment_length; i++)
              segments.push_back(d.segment_descs[i]);
            return segments;
          },
          pybind11::return_value_policy::reference);

  pybind11::class_<air_segment_desc_t>(m, "SegmentDescriptor")
      .def(
          "getHerds",
          [](const air_segment_desc_t &d) -> std::vector<air_herd_desc_t *> {
            std::vector<air_herd_desc_t *> herds;
            for (uint64_t i = 0; i < d.herd_length; i++)
              herds.push_back(d.herd_descs[i]);
            return herds;
          },
          pybind11::return_value_policy::reference)
      .def("getName", [](const air_segment_desc_t &d) -> std::string {
        return {d.name, static_cast<size_t>(d.name_length)};
      });

  pybind11::class_<air_herd_desc_t>(m, "HerdDescriptor")
      .def("getName", [](const air_herd_desc_t &d) -> std::string {
        return {d.name, static_cast<size_t>(d.name_length)};
      });

  m.def("module_load_from_file",
        [](const std::string &filename, queue_t *q) -> air_module_handle_t {
          return air_module_load_from_file(filename.c_str(), q);
        });

  m.def("module_unload", &air_module_unload);

  m.def("get_module_descriptor", &air_module_get_desc,
        pybind11::return_value_policy::reference);

  pybind11::class_<air_agent_t> Agent(m, "Agent");

  m.def(
      "get_agents",
      []() -> std::vector<air_agent_t> {
        std::vector<air_agent_t> agents;
        air_get_agents(agents);
        return agents;
      },
      pybind11::return_value_policy::reference);

  pybind11::class_<queue_t> Queue(m, "Queue");

  m.def(
      "queue_create",
      [](const air_agent_t &a) -> queue_t * {
        queue_t *q = nullptr;
        auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                                    a.handle);
        if (ret != 0)
          return nullptr;
        return q;
      },
      pybind11::return_value_policy::reference);

  m.def(
      "read32", [](uint64_t addr) -> uint32_t { return air_read32(addr); },
      pybind11::return_value_policy::copy);

  m.def("write32", [](uint64_t addr, uint32_t val) -> void {
    return air_write32(addr, val);
  });

  m.def(
      "get_tile_addr",
      [](uint32_t col, uint32_t row) -> uint64_t {
        return air_get_tile_addr(col, row);
      },
      pybind11::return_value_policy::copy);
}

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
  defineAIRHostModule(airhost);
}
