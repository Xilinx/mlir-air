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

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define XAIE_NUM_COLS 10

int
main(int argc, char *argv[])
{
  uint64_t col = 6;
  uint64_t row = 0;

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<hsa_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  uint32_t aie_max_queue_size(0);
  hsa_agent_get_info(agents[0], HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                     &aie_max_queue_size);

  std::cout << "Max AIE queue size: " << aie_max_queue_size << std::endl;

  std::vector<hsa_queue_t *> queues;
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
  if (xaie == NULL) {
    std::cout << "Error getting libxaie context" << std::endl;
    return -1;
  }

  //
  // Set up a 1x3 herd starting 7,0
  //
  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t segment_pkt;
  air_packet_segment_init(&segment_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &segment_pkt);

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t shim_pkt;
  air_packet_device_init(&shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &shim_pkt);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

#define DMA_COUNT 512

  // Ascending plus 2 sequence in the tile memory, and toggle the associated lock
  for (int i=0; i<DMA_COUNT; i++) {
    if (i<(DMA_COUNT/2))
      mlir_aie_write_buffer_a(xaie, i, i+2);
    else
      mlir_aie_write_buffer_b(xaie, i-(DMA_COUNT/2), i+2);
  }
  mlir_aie_release_lock(xaie, 6, 2, 0, 0x1, 0);
  mlir_aie_release_lock(xaie, 6, 2, 1, 0x1, 0);

  uint32_t *dram_ptr = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));
  // Lets stomp over it!
  for (int i=0;i<DMA_COUNT;i++) {
    dram_ptr[i] = 0xdeadbeef;
  }

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t read_pkt;
  air_packet_nd_memcpy(&read_pkt, 0, col, 0, 0, 8, 2,
                       reinterpret_cast<uint64_t>(dram_ptr),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &read_pkt);

  uint32_t errs = 0;
  // Let go check the tile memory
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d;
    if (i<(DMA_COUNT/2))
      d = mlir_aie_read_buffer_a(xaie, i);
    else
      d = mlir_aie_read_buffer_b(xaie, i-(DMA_COUNT/2));

    if (d != i+2) {
      printf("ERROR: Tile Memory id %d Expected %08X, got %08X\n", i, i+2, d);
      errs++;
    }
  }
  for (int i=0; i<DMA_COUNT; i++) {
    if (dram_ptr[i] != 2 + i) {
      printf("ERROR: L2 Memory id %d Expected %08X, got %08X\n", i, i + 2,
             dram_ptr[i]);
      errs++;
    }
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);
  air_free(dram_ptr);

  // Shutdown AIR and HSA
  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    errs++;
  }

  if (errs == 0) {
    printf("PASS!\n");
  } else {
    printf("fail %d/%d.\n",DMA_COUNT-errs, DMA_COUNT);
  }

  return 0;
}
