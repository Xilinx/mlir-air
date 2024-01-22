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

template <typename T>
inline void air_write_pkt(hsa_queue_t *q, uint32_t packet_id, T *pkt) {
  reinterpret_cast<T *>(q->base_address)[packet_id] = *pkt;
}

inline hsa_status_t air_get_agents(std::vector<hsa_agent_t> &agents) {
  return hsa_iterate_agents(find_aie, (void *)&agents);
}

uint64_t air_wait_all(std::vector<uint64_t> &signals);

hsa_status_t air_load_airbin(hsa_agent_t *agent, hsa_queue_t *q,
                             const char *filename, uint8_t column,
                             uint32_t device_id = 0);
#endif
