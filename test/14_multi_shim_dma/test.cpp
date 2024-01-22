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

#define XAIE_NUM_COLS 20

int
main(int argc, char *argv[])
{
  uint64_t col = 7;
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

  hsa_queue_t *q = NULL;

  // Creating a queue
  auto queue_create_status =
      hsa_queue_create(agents[0], aie_max_queue_size, HSA_QUEUE_TYPE_SINGLE,
                       nullptr, nullptr, 0, 0, &q);

  if (queue_create_status != HSA_STATUS_SUCCESS) {
    std::cout << "hsa_queue_create failed" << std::endl;
  }

  // Adding to our vector of queues
  std::vector<hsa_queue_t *> queues;
  queues.push_back(q);
  assert(queues.size() > 0 && "No queues were sucesfully created!");

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
  if (xaie == NULL) {
    std::cout << "Error getting libxaie context" << std::endl;
    return -1;
  }

  //
  // Set up a 1x5 herd starting 7,0
  //
  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t segment_pkt;
  air_packet_segment_init(&segment_pkt, 0, col, 1, row, 5);
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

#define DMA_COUNT 32

  // We're going to stamp over the memories
  for (int i=0; i<DMA_COUNT; i++) {
    mlir_aie_write_buffer_buf72_0(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf74_0(xaie, i, 0xfeedf00d);
  }

  uint32_t *dram_ptr_1 = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dram_ptr_2 = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dram_ptr_3 = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dram_ptr_4 = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));

  if (dram_ptr_1 == NULL || dram_ptr_2 == NULL || dram_ptr_3 == NULL ||
      dram_ptr_4 == NULL) {
    std::cout << "Couldn't allocate device memory" << std::endl;
    return -1;
  }

  for (int i=0;i<DMA_COUNT;i++) {
    dram_ptr_1[i] = i;
    dram_ptr_2[i] = i * 2;
    dram_ptr_3[i] = 0xf001ba11;
    dram_ptr_4[i] = 0x00051ade;
  }

  // This starts the copying to the tiles

  // Sending data on shim 18
  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt_a;
  air_packet_nd_memcpy(&pkt_a, 0, 18, 1, 0, 4, 2,
                       reinterpret_cast<uint64_t>(dram_ptr_1),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt_a);

  // Sending data on shim 11
  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt_b;
  air_packet_nd_memcpy(&pkt_b, 0, 11, 1, 0, 4, 2,
                       reinterpret_cast<uint64_t>(dram_ptr_2),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt_b);

  // This completes the copying to the tiles, let's move the pattern back

  // Reading back the data on shim 18
  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt_c;
  air_packet_nd_memcpy(&pkt_c, 0, 18, 0, 0, 4, 2,
                       reinterpret_cast<uint64_t>(dram_ptr_3),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt_c);

  // Reading back the data on shim 11
  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt_d;
  air_packet_nd_memcpy(&pkt_d, 0, 11, 0, 0, 4, 2,
                       reinterpret_cast<uint64_t>(dram_ptr_4),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt_d);

  uint32_t errs = 0;
  // Let go check the tile memory
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_buf72_0(xaie, i);
    if (d != i) {
      printf("ERROR: buf72_0 id %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_buf74_0(xaie, i);
    if (d != i*2) {
      printf("ERROR: buf74_0 id %d Expected %08X, got %08X\n", i, i*2, d);
      errs++;
    }
  }
  // And the BRAM we updated
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = dram_ptr_3[i]; // bram_ptr[2*DMA_COUNT+i];;
    if (d != i) {
      printf("ERROR: buf72_0 copy id %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = dram_ptr_4[i]; // bram_ptr[3*DMA_COUNT+i];;
    if (d != i*2) {
      printf("ERROR: buf74_0 copy id %d Expected %08X, got %08X\n", i, i*2, d);
      errs++;
    }
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);
  air_free(dram_ptr_1);
  air_free(dram_ptr_2);
  air_free(dram_ptr_3);
  air_free(dram_ptr_4);

  // Shutdown AIR and HSA
  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    errs++;
  }

  if (errs == 0) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }
}
