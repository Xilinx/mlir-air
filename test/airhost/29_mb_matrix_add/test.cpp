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

// test configuration
#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

#define NUM_3D (IMAGE_WIDTH / TILE_WIDTH)
#define NUM_4D (IMAGE_HEIGHT / TILE_HEIGHT)

int main(int argc, char *argv[]) {
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

  // mlir_aie_print_dma_status(xaie, 7, 2);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  uint32_t *dram_ptr_1 = (uint32_t *)air_malloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_2 = (uint32_t *)air_malloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_3 = (uint32_t *)air_malloc(IMAGE_SIZE * sizeof(uint32_t));

  if (dram_ptr_1 == NULL || dram_ptr_2 == NULL || dram_ptr_3 == NULL) {
    std::cout << "Couldn't allocate device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < IMAGE_SIZE; i++) {
    dram_ptr_1[i] = i + 1;
    dram_ptr_2[i] = 1;
    dram_ptr_3[i] = 0xdeface;
  }

  // stamp over the aie tiles
  for (int i = 0; i < TILE_SIZE; i++) {
    mlir_aie_write_buffer_ping_a(xaie, i, 0xabba0000 + i);
    mlir_aie_write_buffer_pong_a(xaie, i, 0xdeeded00 + i);
    mlir_aie_write_buffer_ping_b(xaie, i, 0xcafe0000 + i);
    mlir_aie_write_buffer_pong_b(xaie, i, 0xfabcab00 + i);
    mlir_aie_write_buffer_ping_c(xaie, i, 0x12345670 + i);
    mlir_aie_write_buffer_pong_c(xaie, i, 0x76543210 + i);
  }

  //
  // packet to send the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt_a;
  air_packet_nd_memcpy(
      &pkt_a, 0, col, 1, 0, 4, 2, reinterpret_cast<uint64_t>(dram_ptr_1),
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt_a.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt_a);

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt_b;
  air_packet_nd_memcpy(
      &pkt_b, 0, col, 1, 1, 4, 2, reinterpret_cast<uint64_t>(dram_ptr_2),
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt_b.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt_b);

  //
  // packet to read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt_c;
  air_packet_nd_memcpy(
      &pkt_c, 0, col, 0, 0, 4, 2,
      reinterpret_cast<uint64_t>(
          dram_ptr_3) /*BRAM_ADDR+(2*IMAGE_SIZE*sizeof(float))*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // dispatch the packets and destroy the completion signals that we don't need
  //

  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt_c);
  hsa_signal_destroy(pkt_a.completion_signal);
  hsa_signal_destroy(pkt_b.completion_signal);

  int errors = 0;
  // check the aie tiles
  for (int i = 0; i < TILE_SIZE; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_a(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_a(xaie, i);
    uint32_t d4 = mlir_aie_read_buffer_ping_b(xaie, i);
    uint32_t d5 = mlir_aie_read_buffer_pong_b(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_c(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_c(xaie, i);
    if (d0 + d4 != d2) {
      printf("mismatch [%d] ping %x+%x != %x\n", i, d0, d4, d2);
      errors++;
    }
    if (d1 + d5 != d3) {
      printf("mismatch [%d] pong %x+%x != %x\n", i, d1, d5, d3);
      errors++;
    }
  }

  // check the output image
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr_3[i];
    if (d != (i + 2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);
  air_free(dram_ptr_1);
  air_free(dram_ptr_2);

  // Shutdown AIR and HSA
  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    errors++;
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", errors, IMAGE_SIZE + 2 * TILE_SIZE);
    return -1;
  }
}
