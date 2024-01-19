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
#include <string.h>
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

  uint64_t data = -1;
  char vend[8];
  uint32_t q_iter = 0;
  for (auto q : queues) {
    std::cout << std::endl << "Requesting attribute: AIR_AGENT_INFO_CONTROLLER_ID... ";
    air_get_agent_info(&agents[q_iter], q, AIR_AGENT_INFO_CONTROLLER_ID, &data);
    std::cout << "Agent ID is: " << data << std::endl;

    std::cout << "Requesting attribute: AIR_AGENT_INFO_VENDOR_NAME... ";
    air_get_agent_info(&agents[q_iter], q, AIR_AGENT_INFO_VENDOR_NAME, vend);
    std::cout << "Vendor is: " << vend << std::endl;

    std::cout << "Requesting attribute: AIR_AGENT_INFO_L2_MEM_SIZE... ";
    air_get_agent_info(&agents[q_iter], q, AIR_AGENT_INFO_L2_MEM_SIZE, &data);
    std::cout << "L2 size is: " << std::dec << data << "B" << std::endl;

    // We create one queue on each agent, so we can use the same index
    // to iterate over them
    q_iter++;
  }

  // destroying the queues
  for (auto queue : queues) {
    hsa_queue_destroy(queue);
  }

  // Shutdown AIR and HSA
  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    std::cerr << "[ERROR] air_shut_down() failed" << std::endl;
    return -1;
  }

  std::cout << std::endl << "PASS!" << std::endl;
  return 0;
}
