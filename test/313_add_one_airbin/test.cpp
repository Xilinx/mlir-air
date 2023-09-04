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
#include "air_queue.h"
#include "test_library.h"

#include "aie_inc.cpp"

#define XAIE_NUM_COLS 10
#define DMA_COUNT 16

int main(int argc, char *argv[]) {
  int ret;
  uint64_t row = 0;
  uint64_t col = 6;
	uint64_t wr_idx;
	uint64_t packet_id;
  queue_t *q = nullptr;

  printf("air init\n");
  hsa_status_t init_status = air_init();
  printf("air init done\n");

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  printf("get agents\n");
  std::vector<air_agent_t> agents;
  if ((air_get_agents(agents) != HSA_STATUS_SUCCESS) || agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  // create the queue
  ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                         agents[0].handle, 0);
  assert(ret == 0 && "failed to create queue!");

  ret = air_load_airbin(q, "airbin.elf", col);
  if (!ret) {
    printf("Loading airbin failed: %d\n", ret);
  }

  // Initialize the device memory allocator
  if (air_init_dev_mem_allocator(0x8000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

	// initialize input and output buffers
  uint32_t *src = (uint32_t *)air_dev_mem_alloc(DMA_COUNT * sizeof(uint32_t));
  uint32_t *dst = (uint32_t *)air_dev_mem_alloc(DMA_COUNT * sizeof(uint32_t));
  for (int i = 0; i < DMA_COUNT; i++) {
    src[i] = i + 1;
    dst[i] = 0xdeface;
  }

  // send the input and start processing
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, 0, col, 1, 0, 4, 2, air_dev_mem_get_offset(src),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  // read the output
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt2 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2, air_dev_mem_get_offset(dst),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt2);

	// compare the results
	int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t s = src[i];
    uint32_t d = dst[i];
    // printf("src[%d] = 0x%lx\n", i, src[i]);
    // printf("dst[%d] = 0x%lx\n", i, dst[i]);
    if (d != (s + 1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, s);
    }
  }

  if (errors) {
    printf("fail %d/%d.\n", errors, DMA_COUNT);
    return -1;
  }

  printf("PASS!\n");
  return 0;
}
