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

#define DDR_ADDR  0x2000

// test configuration
#define IMAGE_WIDTH 192
#define IMAGE_HEIGHT 192
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
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

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // setup the herd
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *segment_pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_segment_init(segment_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(q, wr_idx, segment_pkt);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(q, wr_idx, shim_pkt);

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();

  mlir_aie_print_dma_status(xaie, 7, 2);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  // Initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x100000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  uint32_t *dram_ptr_1 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_2 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_3 =
      (uint32_t *)air_dev_mem_alloc(IMAGE_SIZE * sizeof(uint32_t));
  if (dram_ptr_1 != NULL && dram_ptr_2 != NULL && dram_ptr_3 != NULL) {
    for (int i=0; i<IMAGE_SIZE; i++) {
      dram_ptr_1[i] = i + 1;
      dram_ptr_2[i] = i + 1;
      dram_ptr_3[i] = 0xdeface;
    }
  } else {
    printf("[ERROR] Was unable to allocate device memory\n");
    return -1;
  }

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
  // packet to read the output matrix
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_c = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_c, 0, col, 0, 0, 4, 2,
      air_dev_mem_get_pa(dram_ptr_3) /*DDR_ADDR+(2*IMAGE_SIZE*sizeof(float))*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // packet to send the input matrices
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_a = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_a, 0, col, 1, 0, 4, 2, air_dev_mem_get_pa(dram_ptr_1) /*DDR_ADDR*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_b = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_b, 0, col, 1, 1, 4, 2,
      air_dev_mem_get_pa(dram_ptr_2) /*DDR_ADDR+(IMAGE_SIZE*sizeof(float))*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // dispatch the packets to the MB
  //

  air_queue_dispatch_and_wait(q, wr_idx-2, pkt_c);
  
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
    if (d != ((i+1)*2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }
  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, IMAGE_SIZE+2*TILE_SIZE);
    return -1;
  }

  air_dev_mem_allocator_free();
}
