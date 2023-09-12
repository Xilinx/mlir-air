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
  uint64_t col = 7;
  uint64_t row = 0;

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
    auto create_queue_ret =
        air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, agent.handle,
                         0 /* device_id (optional) */);
    assert(create_queue_ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
  if (xaie == NULL) {
    std::cout << "Error initializing libxaie" << std::endl;
    return -1;
  }

  // Want to initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x8000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(queues[0], wr_idx, shim_pkt);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *segment_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_segment_init(segment_pkt, 0, col, 1, row, 5);
  air_queue_dispatch_and_wait(queues[0], wr_idx, segment_pkt);

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

  uint32_t *dram_ptr_1 =
      (uint32_t *)air_dev_mem_alloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dram_ptr_2 =
      (uint32_t *)air_dev_mem_alloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dram_ptr_3 =
      (uint32_t *)air_dev_mem_alloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dram_ptr_4 =
      (uint32_t *)air_dev_mem_alloc(DMA_COUNT * sizeof(uint32_t));

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

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, col, 1, 0, 4, 2,
                       air_dev_mem_get_pa(dram_ptr_1) /*BRAM_ADDR*/,
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_a);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_b =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_b, 0, col, 1, 1, 4, 2,
      air_dev_mem_get_pa(dram_ptr_2) /*BRAM_ADDR+(DMA_COUNT*sizeof(float))*/,
      DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_b);

  // This completes the copying to the tiles, let's move the pattern back

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_c, 0, col, 0, 0, 4, 2,
      air_dev_mem_get_pa(dram_ptr_3) /*BRAM_ADDR+(2*DMA_COUNT*sizeof(float))*/,
      DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_c);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_d =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_d, 0, col, 0, 1, 4, 2,
      air_dev_mem_get_pa(dram_ptr_4) /*BRAM_ADDR+(3*DMA_COUNT*sizeof(float))*/,
      DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_d);

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
  // And the DRAM we updated
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = dram_ptr_3[i] /*bram_ptr[2*DMA_COUNT+i]*/;
    if (d != i) {
      printf("ERROR: dram_ptr_3 copy id %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = dram_ptr_4[i] /*bram_ptr[3*DMA_COUNT+i]*/;
    if (d != i*2) {
      printf("ERROR: dram_ptr_4 copy id %d Expected %08X, got %08X\n", i, i * 2,
             d);
      errs++;
    }
  }

  air_dev_mem_allocator_free();

  if (errs == 0) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }

}
