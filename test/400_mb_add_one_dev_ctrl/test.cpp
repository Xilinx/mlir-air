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
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "air.hpp"
#include "test_library.h"

#include "aie_inc.cpp"

int
main(int argc, char *argv[])
{
  uint64_t row = 0;
  uint64_t col = 6;

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

  if(agents.size() < 2) {
    std::cout << "[ERROR] Need at least 2 agents for this test" << std::endl;
  }

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    auto create_queue_ret =
        air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, agent.handle,
                         0 /* device_id (optional) */);
    if(create_queue_ret) {
      printf("Failed to create queue. Not adding to list of queues\n");
    }
    else {
      queues.push_back(q);
    }
  }

  assert(queues.size() > 0 && "No queues were sucesfully created!");

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
  if (xaie == NULL) {
    std::cout << "Error getting libxaie context" << std::endl;
    return -1;
  }

  // Initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x8000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  //
  // Set up a 1x3 herd starting 7,0
  //
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *herd_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(queues[0], wr_idx, herd_pkt);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(queues[0], wr_idx, shim_pkt);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

#define DMA_COUNT 16

  // Allocating some device memory
  uint32_t *src = (uint32_t *)air_dev_mem_alloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dst = (uint32_t *)air_dev_mem_alloc(DMA_COUNT * sizeof(uint32_t));

  if (src == NULL || dst == NULL) {
    std::cout << "Could not allocate src and dst in device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < DMA_COUNT; i++) {
    src[i] = i + 1;
    dst[i] = 0xdeface;
  }

  for (int i=0; i<8; i++) {
    mlir_aie_write_buffer_ping_in(xaie, i, 0xabbaba00+i);
    mlir_aie_write_buffer_pong_in(xaie, i, 0xdeeded00+i);
    mlir_aie_write_buffer_ping_out(xaie, i, 0x12345670+i);
    mlir_aie_write_buffer_pong_out(xaie, i, 0x76543210+i);
  }

  //
  // send the data
  //

  wr_idx = queue_add_write_index(queues[2], 1);
  packet_id = wr_idx % queues[2]->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(queues[2]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, 0, col, 1, 0, 4, 2, air_dev_mem_get_pa(src),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[2], wr_idx, pkt);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(queues[2], 1);
  packet_id = wr_idx % queues[2]->size;
  dispatch_packet_t *pkt2 =
      (dispatch_packet_t *)(queues[2]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2, air_dev_mem_get_pa(dst),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[2], wr_idx, pkt2);

  int errors = 0;

  for (int i=0; i<8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out(xaie, i);
    if (d0+1 != d2) {
      printf("mismatch ping %x != %x\n", d0, d2);
      errors++;
    } 
    if (d1+1 != d3) {
      printf("mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t s = src[i];
    uint32_t d = dst[i];
    printf("src[%d] = 0x%lx\n", i, src[i]);
    printf("dst[%d] = 0x%lx\n", i, dst[i]);
    if (d != (s + 1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, s);
    }
  }

  // Don't call libxaie deinit so need to free the allocator here
  air_dev_mem_allocator_free();

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, DMA_COUNT);
    return -1;
  }

}
