//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <vector>

#include "air.hpp"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

int main(int argc, char *argv[]) {

  std::vector<hsa_queue_t *> queues;
  uint32_t aie_max_queue_size(0);

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<hsa_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  // Creating a queue on each agent
  for (auto agent : agents) {
    hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                       &aie_max_queue_size);
    std::cout << "Max AIE queue size: " << aie_max_queue_size << std::endl;
    hsa_queue_t *q = NULL;
    auto queue_create_status =
        hsa_queue_create(agent, aie_max_queue_size, HSA_QUEUE_TYPE_SINGLE,
                         nullptr, nullptr, 0, 0, &q);

    if (queue_create_status != HSA_STATUS_SUCCESS) {
      std::cout << "hsa_queue_create failed" << std::endl;
    }

    // Adding to our vector of queues
    queues.push_back(q);
  }

  assert(queues.size() > 0 && "No queues were sucesfully created!");

  // Printing the virtual address of each queue
  for (auto queue : queues) {
    std::cout << "Queue located at vaddr 0x" << queue << std::endl;
  }

  // destroying the queues
  for (auto queue : queues) {
    hsa_queue_destroy(queue);
  }

  // Shutdown AIR and HSA
  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    return -1;
  }

  std::cout << "PASS!" << std::endl;

  return 0;
}
