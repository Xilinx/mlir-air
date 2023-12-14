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

int main(int argc, char *argv[]) {
  uint64_t row = 0;
  uint64_t reset_col = 6;
  uint64_t non_reset_col = 7;

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
  if (xaie == NULL) {
    std::cout << "Error getting libxaie context" << std::endl;
    return -1;
  }

  // Initializing device
  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t shim_pkt;
  air_packet_device_init(&shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &shim_pkt);

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t segment_pkt;
  air_packet_segment_init(&segment_pkt, 0, reset_col, 2, row, 3);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &segment_pkt);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

#define DMA_COUNT 16

  // Allocating some device memory
  uint32_t *src_0 = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dst_0 = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *src_1 = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dst_1 = (uint32_t *)air_malloc(DMA_COUNT * sizeof(uint32_t));

  if (src_0 == NULL || dst_0 == NULL || src_1 == 0 || dst_1 == 0) {
    std::cout << "Could not allocate src and dst in device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < DMA_COUNT; i++) {
    src_0[i] = i + 1;
    src_1[i] = i + 1;
    dst_0[i] = 0xdeface;
    dst_1[i] = 0xdeface;
  }

  for (int i = 0; i < 8; i++) {
    mlir_aie_write_buffer_ping_in(xaie, i, 0xabbaba00 + i);
    mlir_aie_write_buffer_pong_in(xaie, i, 0xdeeded00 + i);
    mlir_aie_write_buffer_ping_out(xaie, i, 0x12345670 + i);
    mlir_aie_write_buffer_pong_out(xaie, i, 0x76543210 + i);
  }

  ///////////// Starting the column we are resetting
  //
  // send the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt;
  air_packet_nd_memcpy(&pkt, 0, reset_col, 1, 0, 4, 2,
                       reinterpret_cast<uint64_t>(src_0),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt);

  //
  // read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt2;
  air_packet_nd_memcpy(&pkt2, 0, reset_col, 0, 0, 4, 2,
                       reinterpret_cast<uint64_t>(dst_0),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt2);

  ///////////// Starting the column we are not resetting
  //
  // send the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt3;
  air_packet_nd_memcpy(&pkt3, 0, non_reset_col, 1, 0, 4, 2,
                       reinterpret_cast<uint64_t>(src_1),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt3);

  //
  // read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt4;
  air_packet_nd_memcpy(&pkt4, 0, non_reset_col, 0, 0, 4, 2,
                       reinterpret_cast<uint64_t>(dst_1),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt4);

  // Reading the status of the two shim DMAs after performing the data movement
  u32 before_reset_core_dma_mm2s_status;
  u32 before_non_reset_core_dma_mm2s_status;
  XAie_Read32(&(xaie->DevInst),
              _XAie_GetTileAddr(&(xaie->DevInst), row, reset_col) + 0x0001D164,
              &before_reset_core_dma_mm2s_status);
  XAie_Read32(&(xaie->DevInst),
              _XAie_GetTileAddr(&(xaie->DevInst), row, non_reset_col) +
                  0x0001D164,
              &before_non_reset_core_dma_mm2s_status);

  // Performing a reset on the reset col
  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t segment_pkt_two;
  air_packet_segment_init(&segment_pkt_two, 0, reset_col, 1, row, 3);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &segment_pkt_two);

  // Reading the status of the two shim DMAs after performing a reset of the
  // reset_col
  u32 after_reset_core_dma_mm2s_status;
  u32 after_non_reset_core_dma_mm2s_status;
  XAie_Read32(&(xaie->DevInst),
              _XAie_GetTileAddr(&(xaie->DevInst), row, reset_col) + 0x0001D164,
              &after_reset_core_dma_mm2s_status);
  XAie_Read32(&(xaie->DevInst),
              _XAie_GetTileAddr(&(xaie->DevInst), row, non_reset_col) +
                  0x0001D164,
              &after_non_reset_core_dma_mm2s_status);

  printf("Before the reset\n");
  printf("(%d, %d)\n", reset_col, row);
  printf("\tMM2S status: 0x%x\n", before_reset_core_dma_mm2s_status);
  printf("(%d, %d)\n", non_reset_col, row);
  printf("\tMM2S status: 0x%x\n", before_non_reset_core_dma_mm2s_status);

  printf("After the reset\n");
  printf("(%d, %d)\n", reset_col, row);
  printf("\tMM2S status: 0x%x\n", after_reset_core_dma_mm2s_status);
  printf("(%d, %d)\n", non_reset_col, row);
  printf("\tMM2S status: 0x%x\n", after_non_reset_core_dma_mm2s_status);

  int errors = 0;

  // Making sure that the core DMA status of both cores are not 0 before the
  // reset - This makes sure the test was run correctly
  if (!before_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of reset core before reset is not zero (0x%x "
           "!= 0)\n",
           before_reset_core_dma_mm2s_status);
  }
  if (!before_non_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of non reset core before reset is not zero "
           "(0x%x != 0)\n",
           before_non_reset_core_dma_mm2s_status);
  }

  // Making sure the reset core DMA status doesn't match what it was before the
  // reset
  if (before_reset_core_dma_mm2s_status == after_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of reset core is the same before (0x%x) and "
           "after (0x%x) the reset\n",
           before_reset_core_dma_mm2s_status, after_reset_core_dma_mm2s_status);
  }

  // Making sure that the reset core DMA status is 0 after the reset
  if (after_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of reset core after reset is not zero (0x%x != "
           "0)\n",
           after_reset_core_dma_mm2s_status);
  }

  // Making sure the non reset core DMA status is the same as what it was before
  // the reset
  if (before_non_reset_core_dma_mm2s_status !=
      after_non_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of non reset core is not the same before "
           "(0x%x) and after (0x%x) the reset\n",
           before_reset_core_dma_mm2s_status, after_reset_core_dma_mm2s_status);
  }

  for (int i = 0; i < 8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out(xaie, i);
    if (d0 + 1 != d2) {
      printf("mismatch ping %x != %x\n", d0, d2);
      errors++;
    }
    if (d1 + 1 != d3) {
      printf("mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t s = src_0[i];
    uint32_t d = dst_0[i];
    if (d != (s + 1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, s);
    }

    s = src_1[i];
    d = dst_1[i];
    if (d != (s + 1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, s);
    }
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);
  air_free(src_0);
  air_free(src_1);
  air_free(dst_0);
  air_free(dst_1);

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
    printf("fail %d.\n", errors);
    return -1;
  }
}
