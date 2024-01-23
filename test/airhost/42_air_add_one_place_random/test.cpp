//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
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

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace air::segments::segment_0 {
int32_t mlir_aie_read_buffer_scratch_0_0(aie_libxaie_ctx_t *, int);
int32_t mlir_aie_read_buffer_scratch_copy_0_0(aie_libxaie_ctx_t *, int);
void mlir_aie_write_buffer_scratch_0_0(aie_libxaie_ctx_t *, int, int32_t);
void mlir_aie_write_buffer_scratch_copy_0_0(aie_libxaie_ctx_t *, int, int32_t);
}; // namespace air::segments::segment_0
using namespace air::segments::segment_0;

#define DMA_COUNT 16

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

  for (int i = 0; i < DMA_COUNT; i++) {
    mlir_aie_write_buffer_scratch_0_0(xaie, i, 0xfadefade);
    mlir_aie_write_buffer_scratch_copy_0_0(xaie, i, 0xfadefade);
  }

  printf("loading air module\n");
  auto handle = air_module_load_from_file(nullptr, &agents[0], q);
  assert(handle && "failed to open air module");

  auto graph_fn =
      (void (*)(void *, void *))dlsym((void *)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t, 1> input;
  tensor_t<uint32_t, 1> output;

  input.shape[0] = DMA_COUNT;
  input.alloc = input.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input.shape[0]);

  output.shape[0] = DMA_COUNT;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0]);

  for (int i = 0; i < DMA_COUNT; i++) {
    input.data[i] = i + 0x1;
    output.data[i] = 0x00defaced;
  }

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  int errors = 0;
  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t d0 = mlir_aie_read_buffer_scratch_0_0(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_scratch_copy_0_0(xaie, i);
    if (d0 + 1 != d1) {
      printf("mismatch tile %x != %x\n", d0, d1);
      errors++;
    }
  }

  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t d = output.data[i];
    if (d != (i + 2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  // Clean up
  free(input.alloc);
  free(output.alloc);
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
    printf("fail %d/%d.\n", errors, 2 * DMA_COUNT);
    return -1;
  }
}
