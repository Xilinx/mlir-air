//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>  // uint64_t
#include <cstdio>   // printf
#include <cstdlib>  // atoi
#include <ctime>    // clock_gettime
#include <fcntl.h>  // open
#include <fstream>  // ifstream
#include <iomanip>  // setw, dec, hex
#include <iostream> // cout
#include <string>
#include <sys/mman.h>
#include <sys/time.h>
#include <vector>

#include "air.hpp"

int addone_driver(queue_t *q, int col, int row) {
  /////////////////////////////////////////////////////////////////////////////
  //////////////////////// Run Add One Application ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

#define DMA_COUNT 16

  // setup images in memory
  uint32_t *bb_ptr;
  uint64_t paddr = 0x2000;

  // Initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x8000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  bb_ptr = (uint32_t *)air_dev_mem_alloc(0x100000);
  paddr = air_dev_mem_get_pa(bb_ptr);
  for (int i = 0; i < DMA_COUNT; i++) {
    bb_ptr[i] = i + 1;
    bb_ptr[DMA_COUNT + i] = 0xdeface;
  }

  //
  // send the data
  //

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt1 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt1, 0, col, 1, 0, 4, 2, paddr,
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt2 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2,
                       paddr + (DMA_COUNT * sizeof(float)),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt2);

  int errors = 0;

  for (uint32_t i = 0; i < DMA_COUNT; i++) {
    volatile uint32_t d = bb_ptr[DMA_COUNT + i];
    if (d != (i + 2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  air_dev_mem_allocator_free();

  if (!errors) {
    return 0;
  } else {
    return 1;
  }
}

int matadd_driver(queue_t *q, int col, int row) {
// test configuration
#define IMAGE_WIDTH 192
#define IMAGE_HEIGHT 192
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

#define NUM_3D (IMAGE_WIDTH / TILE_WIDTH)
#define NUM_4D (IMAGE_HEIGHT / TILE_HEIGHT)

  // setup images in memory
  uint32_t *dram_ptr;
  uint64_t dram_paddr = 0x2000;

  // Initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x8000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  dram_ptr = (uint32_t *)air_dev_mem_alloc(0x100000);
  dram_paddr = air_dev_mem_get_pa(dram_ptr);
  if (dram_ptr != NULL) {
    if ((3 * IMAGE_SIZE * sizeof(uint32_t)) > 0x100000) {
      printf("Image buffers out of range!\n");
      return -1;
    }
    for (int i = 0; i < IMAGE_SIZE; i++) {
      dram_ptr[i] = i + 1;
      dram_ptr[IMAGE_SIZE + i] = i + 1;
      dram_ptr[2 * IMAGE_SIZE + i] = 0xdeface;
    }
  } else
    return -1;

  //
  // packet to send the input matrices
  //

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, col, 1, 0, 4, 2, dram_paddr,
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_b =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_b, 0, col, 1, 1, 4, 2, dram_paddr + (IMAGE_SIZE * sizeof(float)),
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // packet to read the output matrix
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_c, 0, col, 0, 0, 4, 2, dram_paddr + (2 * IMAGE_SIZE * sizeof(float)),
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  //
  // dispatch the packets to the MB
  //

  air_queue_dispatch_and_wait(q, wr_idx, pkt_c);

  int errors = 0;

  // check the output image
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr[2 * IMAGE_SIZE + i];
    if (d != ((i + 1) + (i + 1))) {
      errors++;
      printf("mismatch %x != %x\n", d, 2 * (i + 1));
    }
  }

  air_dev_mem_allocator_free();

  if (!errors) {
    return 0;
  } else {
    return 1;
  }
}

int main(int argc, char **argv) {
  std::string airbin_name_1 = "addone.airbin";
  std::string airbin_name_2 = "matadd.airbin";

  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.size() < 2) {
    std::cout << "failed to enumerate at least 2 agents." << std::endl;
    return -1;
  }

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                           agent.handle);
    assert(ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  if (air_load_airbin(queues[0], airbin_name_1.c_str(), 7) != 0) {
    std::cout << "Error loading airbin 1" << std::endl;
    return 1;
  }
  if (air_load_airbin(queues[1], airbin_name_2.c_str(), 7) != 0) {
    std::cout << "Error loading airbin 2" << std::endl;
    return 1;
  }

  int errors = 0;
  errors += addone_driver(queues[0], 6, 2);
  errors += matadd_driver(queues[1], 7, 2);

  if (!errors) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "fail." << std::endl;
    return -1;
  }
}
