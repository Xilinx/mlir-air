// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <vector>

#include "air_host.h"
#include "acdc_queue.h"
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
hsa_status_t air_get_agents(void *data);

int main(int argc, char *argv[])
{
  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(&agents);
  assert(ret == 0 && "failed to enumerate agents!");

  if (agents.empty()) {
    printf("fail.\n");
    return -1;
  }
 
  for (auto agent : agents) {
    printf("Creating queue using address found at paddr %p\n",agent);
    // create the queue
    queue_t *q = nullptr;
    ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, agent.handle);
    assert(ret == 0 && "failed to create queue!");
  }

  printf("PASS!\n");
  return 0;
}

//
// Implementation to absorb into acdc_queue.h/queue.cpp
//
hsa_status_t air_get_agents(void *data) {
  std::vector<air_agent_t>* pAgents = nullptr;

  if (data == nullptr) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  } else {
      pAgents = static_cast<std::vector<air_agent_t>*>(data);
  }

  uint64_t total_controllers = 0;

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR;

  uint64_t *bram_base = (uint64_t *)mmap(NULL, 0x1000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE);

  total_controllers = bram_base[65];
  if (total_controllers < 1)
    return HSA_STATUS_ERROR;

  uint64_t* base_addr = (uint64_t*)AIR_VCK190_SHMEM_BASE;
  for (int i=0; i<total_controllers; i++) {
    air_agent_t a;
    a.handle = (uint64_t)&base_addr[i];
    pAgents->push_back(a);
  }

  return HSA_STATUS_SUCCESS;
}

