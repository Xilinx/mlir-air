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

#define XAIE_NUM_COLS 36

int
main(int argc, char *argv[])
{
  uint64_t row  = 0;
  uint64_t col  = 7;
  uint64_t col2 = 34;

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

  if (queues.size() < 2) {
    std::cout << "WARNING: test requires at least 2 queues, exiting."
              << std::endl;
    return 0;
  }

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();
  if (xaie == NULL) {
    std::cout << "Error initializing libxaie" << std::endl;
    return -1;
  }

  //
  // Set up a 1x3 herd starting 7,0
  //
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *segment_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_segment_init(segment_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(queues[0], wr_idx, segment_pkt);
  //
  // Set up a 1x3 herd starting 34,0
  //
  uint64_t wr_idx2 = queue_add_write_index(queues[1], 1);
  uint64_t packet_id2 = wr_idx2 % queues[1]->size;
  dispatch_packet_t *segment_pkt2 =
      (dispatch_packet_t *)(queues[1]->base_address_vaddr) + packet_id2;
  air_packet_segment_init(segment_pkt2, 0, col2, 1, row, 3);
  air_queue_dispatch_and_wait(queues[1], wr_idx2, segment_pkt2);

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

  // Want to initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x8000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

#define DMA_COUNT 16

  uint32_t *bram_ptr =
      (uint32_t *)air_dev_mem_alloc(4 * DMA_COUNT * sizeof(uint32_t));
  if (bram_ptr != NULL) {
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = i+1;
      bram_ptr[DMA_COUNT+i]   = 0xdeface;
      bram_ptr[2*DMA_COUNT+i] = 0xdeface;
    }
  }

  for (int i=0; i<8; i++) {
    mlir_aie_write_buffer_ping_in(xaie, i, 0xabbaba10+i);
    mlir_aie_write_buffer_pong_in(xaie, i, 0xdeeded10+i);
    mlir_aie_write_buffer_ping_out(xaie, i, 0x12345610+i);
    mlir_aie_write_buffer_pong_out(xaie, i, 0x76543210+i);
    mlir_aie_write_buffer_ping_in2(xaie, i, 0xabbaba20+i);
    mlir_aie_write_buffer_pong_in2(xaie, i, 0xdeeded20+i);
    mlir_aie_write_buffer_ping_out2(xaie, i, 0x12345620+i);
    mlir_aie_write_buffer_pong_out2(xaie, i, 0x76543220+i);
  }

  // create the queues
  queue_t *q = queues[0];
  queue_t *q2 = queues[1];

  //
  // send the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt1 = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt1, 0, 7, 1, 0, 4, 2, air_dev_mem_get_pa(bram_ptr),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt2 = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, 7, 0, 0, 4, 2,
                       air_dev_mem_get_pa(bram_ptr) +
                           (DMA_COUNT * sizeof(float)),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // send the data
  //

  wr_idx2 = queue_add_write_index(q2, 1);
  packet_id2 = wr_idx2 % q2->size;
  dispatch_packet_t *pkt12 = (dispatch_packet_t*)(q2->base_address_vaddr) + packet_id2;
  air_packet_nd_memcpy(pkt12, 0, 34, 1, 0, 4, 2, air_dev_mem_get_pa(bram_ptr),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx2 = queue_add_write_index(q2, 1);
  packet_id2 = wr_idx2 % q2->size;
  dispatch_packet_t *pkt22 = (dispatch_packet_t*)(q2->base_address_vaddr) + packet_id2;
  air_packet_nd_memcpy(pkt22, 0, 34, 0, 0, 4, 2,
                       air_dev_mem_get_pa(bram_ptr) +
                           (2 * DMA_COUNT * sizeof(float)),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);

  air_queue_dispatch(q, wr_idx, pkt2);
  air_queue_dispatch_and_wait(q2, wr_idx2, pkt22);
  air_queue_wait(q, pkt2);

  int errors = 0;

  for (int i=0; i<8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out(xaie, i);
    if (d0+1 != d2) {
      printf("1 mismatch ping %x != %x\n", d0, d2);
      errors++;
    }
    if (d1+1 != d3) {
      printf("1 mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[DMA_COUNT+i];
    if (d != (i+2)) {
      errors++;
      printf("1 mismatch %x != 2 + %x\n", d, i);
    }
  }
  for (int i=0; i<8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in2(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in2(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out2(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out2(xaie, i);
    if (d0+1 != d2) {
      printf("2 mismatch ping %x != %x\n", d0, d2);
      errors++;
    }
    if (d1+1 != d3) {
      printf("2 mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[2*DMA_COUNT+i];
    if (d != (i+2)) {
      errors++;
      printf("2 mismatch %x != 2 + %x\n", d, i);
    }
  }

  air_dev_mem_allocator_free();

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, 2*(DMA_COUNT+4*8));
    return -1;
  }

}
