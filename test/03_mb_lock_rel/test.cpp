//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "air.hpp"
#include "test_library.h"

#include "aie_inc.cpp"

#define XAIE_NUM_COLS 10

int main(int argc, char *argv[])
{
  auto row = 2;
  auto col = 7;
  auto num_rows = 1;
  auto num_cols = 1;

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

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(q, wr_idx, shim_pkt);

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();

  mlir_aie_print_tile_status(xaie, col, row);

  // Run auto generated config functions
  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);

  mlir_aie_release_lock(xaie, col, 2, 0, 1, 0);
  auto lock_ret = mlir_aie_acquire_lock(xaie, col, 2, 0, 1, 10000);
  assert(lock_ret);

  mlir_aie_configure_dmas(xaie);

  // setup the shim dma descriptors
  uint32_t *bram_ptr;
  mlir_aie_init_mems(xaie, 1);
  bram_ptr = (uint32_t *)mlir_aie_mem_alloc(xaie, 0, 0x8000);

  bram_ptr[24] = 0xacdc;
  mlir_aie_sync_mem_dev(xaie, 0);

  #define DMA_COUNT 256

  auto burstlen = 4;
  XAie_DmaDesc dma_bd;
  XAie_DmaDescInit(&(xaie->DevInst), &dma_bd, XAie_TileLoc(col,0));
  XAie_DmaSetAddrLen(&dma_bd, (u64)bram_ptr, sizeof(u32) * DMA_COUNT); 
  XAie_DmaSetNextBd(&dma_bd, 1, XAIE_DISABLE); 
  XAie_DmaSetAxi(&dma_bd, 0, burstlen, 0, 0, XAIE_ENABLE);
  XAie_DmaEnableBd(&dma_bd);
  XAie_DmaWriteBd(&(xaie->DevInst), &dma_bd, XAie_TileLoc(col,0), 1);
  XAie_DmaChannelPushBdToQueue(&(xaie->DevInst), XAie_TileLoc(col,0), 0, DMA_MM2S, 1);

  u8 cnt = 0;
  XAie_DmaGetPendingBdCount(&(xaie->DevInst), XAie_TileLoc(col,0), 0, DMA_MM2S, &cnt);
  if (cnt)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, cnt);

  XAie_DmaChannelEnable(&(xaie->DevInst), XAie_TileLoc(col,0), 0, DMA_MM2S);

  uint32_t herd_id = 0;
  uint32_t lock_id = 0;

  // reserve a packet in the queue
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  // Set up the worlds smallest herd at 7,2
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, herd_id, col, num_cols, row, num_rows);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // reserve another packet in the queue
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  // lock packet
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_aie_lock(lock_pkt, herd_id, lock_id, /*acq_rel*/1, /*value*/0, 0, 0);
  air_queue_dispatch_and_wait(q, wr_idx, lock_pkt);

  //mlir_aie_release_lock(xaie, col, 2, 0, 0, 0);

  // wait for shim dma to finish
  auto count = 0;
  XAie_DmaGetPendingBdCount(&(xaie->DevInst), XAie_TileLoc(col,0), 0, DMA_MM2S, &cnt);
  while (cnt) {
    sleep(1);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
    XAie_DmaGetPendingBdCount(&(xaie->DevInst), XAie_TileLoc(col,0), 0, DMA_MM2S, &cnt);
  }

  // we copied the start of the shared bram into tile memory,
  // fish out the queue id and check it
  uint32_t d = mlir_aie_read_buffer_b0(xaie, 24);
  printf("ID %x\n", d);

  if (d == 0xacdc) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail.\n");
    return -1;
  }
}
