//===- air_network.h --------------------------------------------*- C++-*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_NETWORK_H
#define AIR_NETWORK_H

#include "air_queue.h"
#include "air_tensor.h"
#include "hsa/hsa.h"

#include <map>
#include <stdint.h>
#include <string>

struct tensor_to_qp_map_entry {
  uint32_t qp;
  uint32_t rkey;
  uint64_t vaddr;
  struct pcie_ernic_buff *local_buff;
};

struct world_view_entry {
  char ip[9];
  char mac[17];
  uint32_t rank;
  uint32_t qps[128];
};

// RDMA operations and variables
hsa_status_t air_set_hostname(char hostname[100]);
hsa_status_t air_get_hostname(char hostname[100]);
hsa_status_t
air_explore_world(uint32_t ernic_id, uint64_t dev_mem_offset,
                  uint64_t bar_offset,
                  std::map<std::string, world_view_entry *> pass_world_view,
                  std::map<std::string, std::string> pass_data_placement);
hsa_status_t air_ernic_free();
hsa_status_t air_ernic_mem_alloc(char buff_name[100], uint32_t size, void *t,
                                 bool register_mem);

void air_recv(hsa_signal_t *s, tensor_t<uint32_t, 1> *t, uint32_t size,
              uint32_t offset, uint32_t src_rank, hsa_agent_t *agent,
              hsa_queue_t *q, uint8_t ernic_sel);
void air_send(hsa_signal_t *s, tensor_t<uint32_t, 1> *t, uint32_t size,
              uint32_t offset, uint32_t dst_rank, hsa_agent_t *agent,
              hsa_queue_t *q, uint8_t ernic_sel);
void air_barrier(tensor_t<uint32_t, 1> *dummy_tensor, hsa_agent_t *agent,
                 hsa_queue_t *q, uint8_t ernic_sel);

#endif
