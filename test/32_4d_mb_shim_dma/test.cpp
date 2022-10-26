//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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
#include <xaiengine.h>

#include "acdc_queue.h"
#include "air_host.h"
#include "hsa_defs.h"

#include "aie_inc.cpp"

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

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_get_agents(&agents);
  assert(get_agents_ret == 0 && "failed to get agents!");

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

  aie_libxaie_ctx_t *xaie = air_init_libxaie();
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
    std::cout << "Couldn't allocate device memory" << std::endl;
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
                       IMAGE_WIDTH * sizeof(float), NUM_3D,
                       TILE_WIDTH * sizeof(float), NUM_4D,
                       IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));

  // Sending the packet to read from the tiles
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(
      pkt_c, 0, col, 0, 0, 4, 2,
      air_dev_mem_get_pa(
          dram_ptr_2) /*AIR_BBUFF_BASE+(IMAGE_SIZE*sizeof(float))*/,
      TILE_WIDTH * sizeof(float), TILE_HEIGHT, IMAGE_WIDTH * sizeof(float),
      NUM_3D, TILE_WIDTH * sizeof(float), NUM_4D,
      IMAGE_WIDTH * TILE_HEIGHT * sizeof(float));
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_c);

  uint32_t errs = 0;

  // Now check the BRAM we updated
  for (int i=0; i<IMAGE_SIZE; i++) {
    uint32_t d = dram_ptr_2[i];
    ;
    if (d != i) {
      printf("ERROR: buf72_0 copy idx %d Expected %08X, got %08X\n", i, i, d);
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
