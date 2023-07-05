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

// test configuration
#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

#define NUM_3D (IMAGE_WIDTH / TILE_WIDTH)
#define NUM_4D (IMAGE_HEIGHT / TILE_HEIGHT)

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

  // setup the herd
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *herd_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(queues[0], wr_idx, herd_pkt);

  // mlir_aie_print_dma_status(xaie, 7, 2);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  // int fd = open("/dev/mem", O_RDWR | O_SYNC);
  // if (fd != -1) {
  // bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED,
  // fd, BRAM_ADDR);
  uint32_t *dram_ptr_1 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_2 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_3 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));

  if (dram_ptr_1 == NULL || dram_ptr_2 == NULL || dram_ptr_3 == NULL) {
    std::cout << "Couldn't allocate device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < IMAGE_SIZE; i++) {
    dram_ptr_1[i] = i + 1;
    dram_ptr_2[i] = 1;
    dram_ptr_3[i] = 0xdeface;
  }
  //} else return -1;

  printf("Eddie Debug:\n");
  printf("dram_ptr_1\tVA: %p\tPA: 0x%lx\n", dram_ptr_1,
         air_dev_mem_get_pa(dram_ptr_1));
  printf("dram_ptr_2\tVA: %p\tPA: 0x%lx\n", dram_ptr_2,
         air_dev_mem_get_pa(dram_ptr_2));
  printf("dram_ptr_3\tVA: %p\tPA: 0x%lx\n", dram_ptr_3,
         air_dev_mem_get_pa(dram_ptr_3));

  // stamp over the aie tiles
  for (int i=0; i<TILE_SIZE; i++) {
    mlir_aie_write_buffer_ping_a(xaie, i, 0xabba0000+i);
    mlir_aie_write_buffer_pong_a(xaie, i, 0xdeeded00+i);
    mlir_aie_write_buffer_ping_b(xaie, i, 0xcafe0000+i);
    mlir_aie_write_buffer_pong_b(xaie, i, 0xfabcab00+i);
    mlir_aie_write_buffer_ping_c(xaie, i, 0x12345670+i);
    mlir_aie_write_buffer_pong_c(xaie, i, 0x76543210+i);
  }

  //
  // packet to read the data
  //

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_c, 0, col, 0, 0, 4, 2,
      air_dev_mem_get_pa(dram_ptr_3) /*BRAM_ADDR+(2*IMAGE_SIZE*sizeof(float))*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // packet to send the data
  //

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_a, 0, col, 1, 0, 4, 2, air_dev_mem_get_pa(dram_ptr_1) /*BRAM_ADDR*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_b =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_b, 0, col, 1, 1, 4, 2,
      air_dev_mem_get_pa(dram_ptr_2) /*BRAM_ADDR+(IMAGE_SIZE*sizeof(float))*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // dispatch the packets to the MB
  //

  air_queue_dispatch_and_wait(queues[0], wr_idx - 2, pkt_c);

  int errors = 0;
  // check the aie tiles
  for (int i=0; i<TILE_SIZE; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_a(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_a(xaie, i);
    uint32_t d4 = mlir_aie_read_buffer_ping_b(xaie, i);
    uint32_t d5 = mlir_aie_read_buffer_pong_b(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_c(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_c(xaie, i);
    if (d0+d4 != d2) {
      printf("mismatch [%d] ping %x+%x != %x\n", i, d0, d4, d2);
      errors++;
    }
    if (d1+d5 != d3) {
      printf("mismatch [%d] pong %x+%x != %x\n", i, d1, d5, d3);
      errors++;
    }
  }

  // check the output image
  for (int i=0; i<IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr_3[i];
    if (d != (i+2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  air_dev_mem_allocator_free();

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, IMAGE_SIZE+2*TILE_SIZE);
    return -1;
  }

}
