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
