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

#define IMAGE_WIDTH 96
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

#define NUM_3D (IMAGE_WIDTH / TILE_WIDTH)
#define NUM_4D (IMAGE_HEIGHT / TILE_HEIGHT)

#define VERBOSE 0

int main(int argc, char *argv[]) {
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
  air_packet_segment_init(segment_pkt, 0, col, 2, row, 5);
  air_queue_dispatch_and_wait(queues[0], wr_idx, segment_pkt);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  // We're going to stamp over the memories
  for (int i = 0; i < 2 * TILE_SIZE; i++) {
    mlir_aie_write_buffer_buf72_0(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf72_1(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf74_0(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf74_1(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf82_0(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf82_1(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf84_0(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf84_1(xaie, i, 0xfeedface);
  }

  uint32_t *dram_ptr_1 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_2 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_3 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_4 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_5 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));

  if (dram_ptr_1 == NULL || dram_ptr_2 == NULL || dram_ptr_3 == NULL ||
      dram_ptr_4 == NULL || dram_ptr_5 == NULL) {
    std::cout << "Couldn't allocate device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < IMAGE_SIZE; i++) {
    /*bram_ptr[i] = i;
    bram_ptr[i+IMAGE_SIZE]   = 0xba110001;
    bram_ptr[i+2*IMAGE_SIZE] = 0xba110002;
    bram_ptr[i+3*IMAGE_SIZE] = 0xba110003;
    bram_ptr[i+4*IMAGE_SIZE] = 0xba110004;*/
    dram_ptr_1[i] = i;
    dram_ptr_2[i] = 0xba110001;
    dram_ptr_3[i] = 0xba110002;
    dram_ptr_4[i] = 0xba110003;
    dram_ptr_5[i] = 0xba110004;
  }

  // Core 7,2
  // Send the packet to write the tiles
  if (VERBOSE)
    printf("[DEBUG] Sending pkt_a\n");
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, 7, 1, 0, 4, 2, air_dev_mem_get_pa(dram_ptr_1),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  // air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_a);

  // Core 7,4
  // Send the packet to write the tiles
  if (VERBOSE)
    printf("[DEBUG] Sending pkt_b\n");
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_b =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_b, 0, 7, 1, 1, 4, 2, air_dev_mem_get_pa(dram_ptr_1),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  // air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_b);

  // Core 8,2
  // Send the packet to write the tiles
  if (VERBOSE)
    printf("[DEBUG] Sending pkt_a\n");
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_e =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_e, 0, 6, 1, 0, 4, 2, air_dev_mem_get_pa(dram_ptr_1),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  // air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_e);

  // Core 8,4
  // Send the packet to write the tiles
  if (VERBOSE)
    printf("[DEBUG] Sending pkt_f\n");
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_f =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_f, 0, 6, 1, 1, 4, 2, air_dev_mem_get_pa(dram_ptr_1),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  // air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_f);

  // Core 7,2
  // Start by sending the packet to read from the tiles
  if (VERBOSE)
    printf("[DEBUG] Sending pkt_c\n");
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_c, 0, 7, 0, 0, 4, 2, air_dev_mem_get_pa(dram_ptr_2),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  // air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_c);

  // Core 7,4
  // Start by sending the packet to read from the tiles
  if (VERBOSE)
    printf("[DEBUG] Sending pkt_d\n");
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_d =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_d, 0, 7, 0, 1, 4, 2, air_dev_mem_get_pa(dram_ptr_3),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  // air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_d);

  // Core 8,2
  // Start by sending the packet to read from the tiles
  if (VERBOSE)
    printf("[DEBUG] Sending pkt_g\n");
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_g =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_g, 0, 6, 0, 0, 4, 2, air_dev_mem_get_pa(dram_ptr_4),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  // air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_g);

  // Core 8,4
  // Start by sending the packet to read from the tiles
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_h =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_h, 0, 6, 0, 1, 4, 2, air_dev_mem_get_pa(dram_ptr_5),
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_h);

  uint32_t errs = 0;
  // Check the BRAM we updated
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr_2[i]; // bram_ptr[IMAGE_SIZE+i];;
    u32 r = i / IMAGE_WIDTH;
    u32 c = i % IMAGE_WIDTH;
    if (d != i) {
      printf("ERROR: buf72_0 copy idx %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr_3[i]; // bram_ptr[2*IMAGE_SIZE+i];;
    u32 r = i / IMAGE_WIDTH;
    u32 c = i % IMAGE_WIDTH;
    if (d != i) {
      printf("ERROR: buf74_0 copy idx %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr_4[i]; // bram_ptr[3*IMAGE_SIZE+i];;
    u32 r = i / IMAGE_WIDTH;
    u32 c = i % IMAGE_WIDTH;
    if (d != i) {
      printf("ERROR: buf82_0 copy idx %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr_5[i]; // bram_ptr[4*IMAGE_SIZE+i];;
    u32 r = i / IMAGE_WIDTH;
    u32 c = i % IMAGE_WIDTH;
    if (d != i) {
      printf("ERROR: buf84_0 copy idx %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }

  air_dev_mem_allocator_free();

  if (errs == 0) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail.\n");
    return -1;
  }
}
