//===- AirHostModule.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "air.hpp"
#include "hsa/hsa.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;

namespace {
void defineAIRHostModule(nb::module_ &m) {

  m.def(
      "init_libxaie", []() -> uint64_t { return (uint64_t)air_init_libxaie(); },
      nb::rv_policy::reference);

  m.def("deinit_libxaie", [](uint64_t ctx) -> void {
    air_deinit_libxaie((air_libxaie_ctx_t)ctx);
  });

  m.def("init", []() -> uint64_t { return (uint64_t)air_init(); });

  m.def("shut_down", []() -> uint64_t { return (uint64_t)air_shut_down(); });

  nb::class_<air_module_desc_t>(m, "ModuleDescriptor")
      .def(
          "getSegments",
          [](const air_module_desc_t &d) -> std::vector<air_segment_desc_t *> {
            std::vector<air_segment_desc_t *> segments;
            for (uint64_t i = 0; i < d.segment_length; i++)
              segments.push_back(d.segment_descs[i]);
            return segments;
          },
          nb::rv_policy::reference);

  nb::class_<air_segment_desc_t>(m, "SegmentDescriptor")
      .def(
          "getHerds",
          [](const air_segment_desc_t &d) -> std::vector<air_herd_desc_t *> {
            std::vector<air_herd_desc_t *> herds;
            for (uint64_t i = 0; i < d.herd_length; i++)
              herds.push_back(d.herd_descs[i]);
            return herds;
          },
          nb::rv_policy::reference)
      .def("getName", [](const air_segment_desc_t &d) -> std::string {
        return {d.name, static_cast<size_t>(d.name_length)};
      });

  nb::class_<air_herd_desc_t>(m, "HerdDescriptor")
      .def("getName", [](const air_herd_desc_t &d) -> std::string {
        return {d.name, static_cast<size_t>(d.name_length)};
      });

  m.def("module_load_from_file",
        [](const std::string &filename, hsa_agent_t *agent,
           hsa_queue_t *q) -> air_module_handle_t {
          return air_module_load_from_file(filename.c_str(), agent, q);
        });

  m.def("module_unload", &air_module_unload);

  m.def("get_module_descriptor", &air_module_get_desc,
        nb::rv_policy::reference);

  nb::class_<hsa_agent_t> Agent(m, "Agent");

  m.def(
      "get_agents",
      []() -> std::vector<hsa_agent_t> {
        std::vector<hsa_agent_t> agents;
        air_get_agents(agents);
        return agents;
      },
      nb::rv_policy::reference);

  nb::class_<hsa_queue_t> Queue(m, "Queue");

  m.def(
      "queue_create",
      [](const hsa_agent_t &a) -> hsa_queue_t * {
        hsa_queue_t *q = nullptr;
        uint32_t aie_max_queue_size(0);

        // Query the queue size the agent supports
        auto queue_size_ret = hsa_agent_get_info(
            a, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &aie_max_queue_size);
        if (queue_size_ret != HSA_STATUS_SUCCESS)
          return nullptr;

        // Creating the queue
        auto queue_create_ret =
            hsa_queue_create(a, aie_max_queue_size, HSA_QUEUE_TYPE_SINGLE,
                             nullptr, nullptr, 0, 0, &q);

        if (queue_create_ret != 0)
          return nullptr;
        return q;
      },
      nb::rv_policy::reference);

  m.def(
      "read32", [](uint64_t addr) -> uint32_t { return air_read32(addr); },
      nb::rv_policy::copy);

  m.def("write32", [](uint64_t addr, uint32_t val) -> void {
    return air_write32(addr, val);
  });

  m.def(
      "get_tile_addr",
      [](uint32_t col, uint32_t row) -> uint64_t {
        return air_get_tile_addr(col, row);
      },
      nb::rv_policy::copy);
}

} // namespace

NB_MODULE(_airRt, m) {
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
