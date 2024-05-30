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

#include "air.hpp"
#include "test_library.h"

#define VERBOSE 1

#define TENSOR_1D 16
#define TENSOR_2D 4
#define TENSOR_3D 2
#define TENSOR_SIZE (TENSOR_1D * TENSOR_2D * TENSOR_3D)

#define TILE_1D 4
#define TILE_2D 2
#define TILE_3D 2
#define TILE_SIZE (TILE_1D * TILE_2D * TILE_3D)

namespace air::segments::segment_0 {
int32_t mlir_aie_read_buffer_buf0(aie_libxaie_ctx_t *, int);
void mlir_aie_write_buffer_buf0(aie_libxaie_ctx_t *, int, int32_t);
}; // namespace air::segments::segment_0
using namespace air::segments::segment_0;

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

  hsa_agent_get_info(agents[0], HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                     &aie_max_queue_size);

  std::cout << "Max AIE queue size: " << aie_max_queue_size << std::endl;

  hsa_queue_t *q = NULL;

  // Creating a queue
  auto queue_create_status =
      hsa_queue_create(agents[0], aie_max_queue_size, HSA_QUEUE_TYPE_SINGLE,
                       nullptr, nullptr, 0, 0, &q);

  if (queue_create_status != HSA_STATUS_SUCCESS) {
    std::cout << "hsa_queue_create failed" << std::endl;
  }

  // Adding to our vector of queues
  queues.push_back(q);
  assert(queues.size() > 0 && "No queues were sucesfully created!");

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
  for (int i = 0; i < TILE_SIZE; i++)
    mlir_aie_write_buffer_buf0(xaie, i, 0xfadefade);

  printf("loading air module\n");
  auto handle = air_module_load_from_file(nullptr, &agents[0], q);
  assert(handle && "failed to open air module");

  auto graph_fn = (void (*)(void *))dlsym((void *)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t, 3> input;

  input.shape[0] = TENSOR_1D;
  input.shape[1] = TENSOR_2D;
  input.shape[2] = TENSOR_3D;
  input.alloc = input.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input.shape[0] * input.shape[1] * input.shape[2]);

  for (int i = 0; i < TENSOR_SIZE; i++) {
    input.data[i] = i;
  }

  void *i;
  i = &input;
  graph_fn(i);

  int errors = 0;

  // Now look at the image, should have the bottom left filled in
  for (int i = 0; i < TILE_SIZE; i++) {
    uint32_t rb = mlir_aie_read_buffer_buf0(xaie, i);
    // An = Aoffset * ((n / Aincrement) % Awrap)
    // Aoffset = add for each increment
    // Awrap = how many increments before wrapping
    // Aincrement = how many streams before increment
    uint32_t xn = 1 * ((i / 1) % 4);
    uint32_t yn = 16 * ((i / 4) % 2);
    uint32_t zn = 64 * ((i / 8) % 2);
    uint32_t a = xn + yn + zn;
    uint32_t vb = input.data[a];
    if (!(rb == vb)) {
      printf("Tile Mem %d should be %08X, is %08X\n", i, vb, rb);
      errors++;
    }
  }

  // Clean up
  free(input.alloc);
  air_module_unload(handle);
  hsa_queue_destroy(queues[0]);

  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    errors++;
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", (TILE_SIZE + TENSOR_SIZE - errors),
           TILE_SIZE + TENSOR_SIZE);
    return -1;
  }
}
