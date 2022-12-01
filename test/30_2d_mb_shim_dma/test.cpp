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

#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

int
main(int argc, char *argv[])
{
  uint64_t col = 7;
  uint64_t row = 0;

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

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();
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

  // We're going to stamp over the memories
  for (int i=0; i<2*TILE_SIZE; i++) {
    mlir_aie_write_buffer_buf72_0(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf72_1(xaie, i, 0xfeedface);
  }

  uint32_t *dram_ptr_1 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_2 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));

  if (dram_ptr_1 == NULL || dram_ptr_2 == NULL) {
    std::cout << "Can't allocate device memory" << std::endl;
    return -1;
  }

  for (int i=0;i<IMAGE_SIZE;i++) {
    dram_ptr_1[i] = i;
    dram_ptr_2[i] = 0xf001ba11;
  }

  // Send the packet to write to the tiles
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, col, 1, 0, 4, 2,
                       air_dev_mem_get_pa(dram_ptr_1) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), 1, 0, 1, 0);

  // Send the packet to read from the tiles
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_c, 0, col, 0, 0, 4, 2,
      air_dev_mem_get_pa(
          dram_ptr_2) /*AIR_BBUFF_BASE+(IMAGE_SIZE*sizeof(float))*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float), 1,
      0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_c);

  uint32_t errs = 0;
  // Let go check the tile memory
  for (int i=0; i<TILE_SIZE; i++) {
    uint32_t d = mlir_aie_read_buffer_buf72_0(xaie, i);
    u32 row = i / TILE_WIDTH;
    u32 col = i % TILE_WIDTH;
    u32 o_i = row * IMAGE_WIDTH + col;
    if (d != o_i) {
      printf("ERROR: buf72_0 idx %d Expected %08X, got %08X\n", i, o_i, d);
      errs++;
    } 
  }
  // And the BRAM we updated
  for (int i=0; i<IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr_2[i]; // bram_ptr[IMAGE_SIZE+i];;
    u32 r = i / IMAGE_WIDTH;
    u32 c = i % IMAGE_WIDTH;
    if ((r < TILE_HEIGHT) && (c < TILE_WIDTH)) {
      if (d != i) {
        printf("ERROR: buf72_0 copy idx %d Expected %08X, got %08X\n", i, i, d);
        errs++;
      }
    } else {
      if (d != 0xf001ba11) {
        printf("ERROR: buf72_0 copy idx %d Expected %08X, got %08X\n", i, 0xf001ba11, d);
        errs++;
      }
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
