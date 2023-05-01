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

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

namespace air::partitions::partition_0 {
void mlir_aie_write_buffer_scratch_0_0(aie_libxaie_ctx_t*, int, int32_t);
};
using namespace air::partitions::partition_0;

int
main(int argc, char *argv[])
{
  uint64_t row = 3;
  uint64_t col = 3;

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    auto create_queue_ret = air_queue_create(
        MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, agent.handle);

    if(create_queue_ret) {
      printf("Failed to create queue. Not adding to list");
    }
    else {
      queues.push_back(q);
    }
  }

  assert(queues.size() > 0 && "No queues were sucesfully created!");

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();

  queue_t *q = queues[0];

  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(queues[0], wr_idx, shim_pkt);

  for (int i=0; i<TILE_SIZE; i++)
    mlir_aie_write_buffer_scratch_0_0(xaie, i, 0xfadefade);

  // Switching the queue to the BP after the device controller initializes
  // the device
  q = queues[1];

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void *))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t,2> input;
  tensor_t<uint32_t,2> output;

  input.shape[1] = IMAGE_WIDTH;
  input.shape[0] = IMAGE_HEIGHT;
  input.alloc = input.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input.shape[0] * input.shape[1]);

  output.shape[1] = IMAGE_WIDTH;
  output.shape[0] = IMAGE_HEIGHT;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0] * output.shape[1]);

  for (int i=0; i<IMAGE_SIZE; i++) {
    input.data[i] = i + 0x1000;
    output.data[i] = 0x00defaced;
  }

  mlir_aie_print_tile_status(xaie, col, row);
  mlir_aie_print_dma_status(xaie, col, row);

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  int errors = 0;

  // Now look at the image, should have the bottom left filled in
  for (int i=0;i<IMAGE_SIZE;i++) {
    u32 rb = output.data[i];

    u32 row = i / IMAGE_WIDTH;
    u32 col = i % IMAGE_WIDTH;

    if ((row >= TILE_HEIGHT) && (col < TILE_WIDTH)) {
      if (!(rb == 0x1000+i)) {
        printf("IM %d [%d, %d] should be %08X, is %08X\n", i, col, row, i+0x1000, rb);
        errors++;
      }
    }
    else {
      if (rb != 0x00defaced) {
        printf("IM %d [%d, %d] should be 0xdefaced, is %08X\n", i, col, row, rb);
        errors++;
      }
    }
  }

  free(input.alloc);
  free(output.alloc);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (TILE_SIZE+IMAGE_SIZE-errors), TILE_SIZE+IMAGE_SIZE);
    return -1;
  }

}
