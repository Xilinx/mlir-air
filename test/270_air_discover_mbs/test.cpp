// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <assert.h>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <vector>

#include "acdc_queue.h"
#include "air_host.h"
#include "hsa_defs.h"

//
// Mock up of functions to discover/enumerate microblaze
// controllers as "air agents".
//

//
// Opaque handle to an air agent.
// Stores the physical address of the memory location
// storing the agent's queue address.
//
typedef struct air_agent_s {
  uint64_t handle;
} air_agent_t;

//
// Fill a vector with paddrs storing queue locations
//
hsa_status_t air_get_agents(std::vector<air_agent_t> *data);

int main(int argc, char *argv[]) {
  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(&agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    std::cout << "Creating queue using address found at paddr 0x"
              << std::hex << agent.handle << std::endl;
    // create the queue
    queue_t *q = nullptr;
    ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                           agent.handle);
    assert(ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  for (auto queue : queues) {
    std::cout << "Queue located at vaddr 0x" << queue << std::endl;
  }

  std::cout << "PASS!" << std::endl;
  return 0;
}

//
// Implementation to absorb into acdc_queue.h/queue.cpp
//
hsa_status_t air_get_agents(std::vector<air_agent_t> *data) {
  std::vector<air_agent_t> *pAgents = nullptr;

  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  } else {
    pAgents = static_cast<std::vector<air_agent_t> *>(data);
  }

  uint64_t total_controllers = 0;

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR;

  uint64_t *bram_base =
      reinterpret_cast<uint64_t *>(mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE));

  total_controllers = bram_base[65];
  if (total_controllers < 1) {
    std::cerr << "No agents found" << std::endl;
    return HSA_STATUS_ERROR;
  }

  uint64_t *base_addr = reinterpret_cast<uint64_t *>(AIR_VCK190_SHMEM_BASE);
  for (int i = 0; i < total_controllers; i++) {
    air_agent_t a;
    a.handle = reinterpret_cast<uintptr_t>(&base_addr[i]);
    pAgents->push_back(a);
  }

  auto res = munmap(bram_base, 0x1000);
  if (res) {
    std::cerr << "Could not munmap" << std::endl;
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}
