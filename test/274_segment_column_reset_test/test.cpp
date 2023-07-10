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

#define XAIE_NUM_COLS 10

int
main(int argc, char *argv[])
{
  uint64_t row = 2;
  uint64_t reset_col = 6;
  uint64_t non_reset_col = 7;

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
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
    auto create_queue_ret =
        air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, agent.handle,
                         0 /* device_id (optional) */);
    assert(create_queue_ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
  if (xaie == NULL) {
    std::cout << "Error getting libxaie context" << std::endl;
    return -1;
  }

  // Initializing the device
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(queues[0], wr_idx, shim_pkt);

  // Performing our reset on the columns we are using
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *segment_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_segment_init(segment_pkt, 0, reset_col, 2, 0 /* Starting reset at row 0 */, 3);
  air_queue_dispatch_and_wait(queues[0], wr_idx, segment_pkt);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  // Reading the DMA status of the two AIEs that we configured before the reset
  u32 before_reset_core_dma_mm2s_status;
  u32 before_reset_core_dma_s2mm_status;
  u32 before_non_reset_core_dma_mm2s_status;
  u32 before_non_reset_core_dma_s2mm_status;
  XAie_Read32(&(xaie->DevInst), _XAie_GetTileAddr(&(xaie->DevInst), row, reset_col) + 0x0001DF10, &before_reset_core_dma_mm2s_status);
  XAie_Read32(&(xaie->DevInst), _XAie_GetTileAddr(&(xaie->DevInst), row, reset_col) + 0x0001DF00, &before_reset_core_dma_s2mm_status);
  XAie_Read32(&(xaie->DevInst), _XAie_GetTileAddr(&(xaie->DevInst), row, non_reset_col) + 0x0001DF10, &before_non_reset_core_dma_mm2s_status);
  XAie_Read32(&(xaie->DevInst), _XAie_GetTileAddr(&(xaie->DevInst), row, non_reset_col) + 0x0001DF00, &before_non_reset_core_dma_s2mm_status);

  // Resetting the shim and columns in reset_col
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *segment_pkt_two =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_segment_init(segment_pkt_two, 0, reset_col, 1, 0 /* Starting reset at row 0 */, 3);
  air_queue_dispatch_and_wait(queues[0], wr_idx, segment_pkt_two);

  // Reading the DMA status of the two AIEs that we configured after the reset
  u32 after_reset_core_dma_mm2s_status;
  u32 after_reset_core_dma_s2mm_status;
  u32 after_non_reset_core_dma_mm2s_status;
  u32 after_non_reset_core_dma_s2mm_status;
  XAie_Read32(&(xaie->DevInst), _XAie_GetTileAddr(&(xaie->DevInst), row, reset_col) + 0x0001DF10, &after_reset_core_dma_mm2s_status);
  XAie_Read32(&(xaie->DevInst), _XAie_GetTileAddr(&(xaie->DevInst), row, reset_col) + 0x0001DF00, &after_reset_core_dma_s2mm_status);
  XAie_Read32(&(xaie->DevInst), _XAie_GetTileAddr(&(xaie->DevInst), row, non_reset_col) + 0x0001DF10, &after_non_reset_core_dma_mm2s_status);
  XAie_Read32(&(xaie->DevInst), _XAie_GetTileAddr(&(xaie->DevInst), row, non_reset_col) + 0x0001DF00, &after_non_reset_core_dma_s2mm_status);

  // Printing the DMA status before and after the reset
  printf("Before the reset\n");
  printf("(%d, %d)\n", reset_col, row);
  printf("\tMM2S status: 0x%x\n", before_reset_core_dma_mm2s_status);
  printf("\tS2MM status: 0x%x\n", before_reset_core_dma_s2mm_status);
  printf("(%d, %d)\n", non_reset_col, row);
  printf("\tMM2S status: 0x%x\n", before_non_reset_core_dma_mm2s_status);
  printf("\tS2MM status: 0x%x\n", before_non_reset_core_dma_s2mm_status);

  printf("After the reset\n");
  printf("(%d, %d)\n", reset_col, row);
  printf("\tMM2S status: 0x%x\n", after_reset_core_dma_mm2s_status);
  printf("\tS2MM status: 0x%x\n", after_reset_core_dma_s2mm_status);
  printf("(%d, %d)\n", non_reset_col, row);
  printf("\tMM2S status: 0x%x\n", after_non_reset_core_dma_mm2s_status);
  printf("\tS2MM status: 0x%x\n", after_non_reset_core_dma_s2mm_status);

  int errors = 0;

  // Making sure that the core DMA status of both cores are not 0 before the reset - This makes sure the test was run correctly
  if(!before_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of reset core before reset is not zero (0x%x != 0)\n", before_reset_core_dma_mm2s_status);
  }
  if(!before_reset_core_dma_s2mm_status) {
    errors++;
    printf("[ERROR] S2MM Status of reset core before reset is not zero (0x%x != 0)\n", before_reset_core_dma_s2mm_status);
  }
  if(!before_non_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of non reset core before reset is not zero (0x%x != 0)\n", before_non_reset_core_dma_mm2s_status);
  }
  if(!before_non_reset_core_dma_s2mm_status) {
    errors++;
    printf("[ERROR] S2MM Status of non reset core before reset is not zero (0x%x != 0)\n", before_non_reset_core_dma_s2mm_status);
  }

  // Making sure the reset core DMA status doesn't match what it was before the reset
  if(before_reset_core_dma_mm2s_status == after_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of reset core is the same before (0x%x) and after (0x%x) the reset\n", before_reset_core_dma_mm2s_status, after_reset_core_dma_mm2s_status);
  }
  if(before_reset_core_dma_s2mm_status == after_reset_core_dma_s2mm_status) {
    errors++;
    printf("[ERROR] S2MM Status of reset core is the same before (0x%x) and after (0x%x) the reset\n", before_reset_core_dma_s2mm_status, after_reset_core_dma_s2mm_status);
  }

  // Making sure that the reset core DMA status is 0 after the reset
  if(after_reset_core_dma_s2mm_status) {
    errors++;
    printf("[ERROR] MM2S Status of reset core after reset is not zero (0x%x != 0)\n", after_reset_core_dma_mm2s_status);
  }
  if(after_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] S2MM Status of reset core after reset is not zero (0x%x != 0)\n", after_reset_core_dma_s2mm_status);
  }

  // Making sure the non reset core DMA status is the same as what it was before the reset
  if(before_non_reset_core_dma_mm2s_status != after_non_reset_core_dma_mm2s_status) {
    errors++;
    printf("[ERROR] MM2S Status of non reset core is not the same before (0x%x) and after (0x%x) the reset\n", before_non_reset_core_dma_mm2s_status, after_non_reset_core_dma_mm2s_status);
  }
  if(before_non_reset_core_dma_s2mm_status != after_non_reset_core_dma_s2mm_status) {
    errors++;
    printf("[ERROR] S2MM Status of non reset core is not the same before (0x%x) and after (0x%x) the reset\n", before_non_reset_core_dma_s2mm_status, after_non_reset_core_dma_s2mm_status);
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d.\n", errors);
    return -1;
  }

}
