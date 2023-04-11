//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "air_host.h"
#include "test_library.h"

#define DMA_COUNT 256

namespace air::segments::segment_0 {
void mlir_aie_write_buffer_buf0(aie_libxaie_ctx_t*, int, int32_t);
};
using namespace air::segments::segment_0;

int
main(int argc, char *argv[])
{
  uint64_t row = 4;
  uint64_t col = 5;

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_iterate_agents(
      [](air_agent_t a, void *d) {
        auto *v = static_cast<std::vector<air_agent_t> *>(d);
        v->push_back(a);
        return HSA_STATUS_SUCCESS;
      },
      (void *)&agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    auto create_queue_ret = air_queue_create(
        MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, agent.handle);
    assert(create_queue_ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();

  queue_t *q = queues[0];

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> output;
  output.shape[0] = DMA_COUNT;
  output.alloc = output.data = (uint32_t *)malloc(sizeof(uint32_t) * DMA_COUNT);
  for (int i=0; i<output.shape[0]; i++) {
    output.data[i] = i;
  }

  auto o = &output;
  graph_fn(o);

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = output.data[i];
    if (d != (i+0x10)) {
      errors++;
      printf("mismatch %x != 0x10 + %x\n", d, i);
    }
  }

  free(output.alloc);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT-errors), DMA_COUNT);
    return -1;
  }

}
