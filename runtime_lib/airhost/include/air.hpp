//===- air.hpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_HPP
#define AIR_HPP

#include "air_host.h"

#include <stdint.h>
#include <vector>

inline hsa_status_t air_get_agents(std::vector<air_agent_t> &agents) {
  return air_iterate_agents(
      [](air_agent_t a, void *d) {
        auto *v = static_cast<std::vector<air_agent_t> *>(d);
        v->push_back(a);
        return HSA_STATUS_SUCCESS;
      },
      (void *)&agents);
}

uint64_t air_wait_all(std::vector<uint64_t> &signals);

int air_load_airbin(queue_t *q, const char *filename, uint8_t column,
                    uint32_t device_id = 0);
#endif
