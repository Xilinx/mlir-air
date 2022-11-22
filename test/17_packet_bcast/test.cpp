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

#include "air_host.h"
#include "test_library.h"

#include "aie_inc.cpp"

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

int main(int argc, char *argv[]) {
  uint64_t col = 7;
  uint64_t row = 2;

  auto init_ret = air_init();
  assert(init_ret == HSA_STATUS_SUCCESS);

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_iterate_agents(
      [](air_agent_t a, void *d) {
        auto *v = static_cast<std::vector<air_agent_t> *>(d);
        v->push_back(a);
        return HSA_STATUS_SUCCESS;
      },
      (void *)&agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && agents.size() &&
         "failed to get agents!");

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
  dispatch_packet_t *herd_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 2, row, 2);
  air_queue_dispatch_and_wait(queues[0], wr_idx, herd_pkt);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(queues[0], wr_idx, shim_pkt);

  XAie_Finish(&(xaie->DevInst));
  XAie_CfgInitialize(&(xaie->DevInst), &(xaie->AieConfigPtr));
  XAie_PmRequestTiles(&(xaie->DevInst), NULL, 0);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  // overwrite the tile memory buffers
  for (int i = 0; i < TILE_SIZE; i++) {
    mlir_aie_write_buffer_buf0(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf1(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf2(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf3(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf4(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf5(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf6(xaie, i, 0xdeadbeef);
    mlir_aie_write_buffer_buf7(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf8(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf9(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf10(xaie, i, 0xfeedface);
    mlir_aie_write_buffer_buf11(xaie, i, 0xfeedface);
  }

  uint32_t *dram_ptr_1 =
      (uint32_t *)air_dev_mem_alloc(TILE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_2 =
      (uint32_t *)air_dev_mem_alloc(TILE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_3 =
      (uint32_t *)air_dev_mem_alloc(TILE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_4 =
      (uint32_t *)air_dev_mem_alloc(TILE_SIZE * sizeof(uint32_t));
  uint32_t *dram_ptr_5 =
      (uint32_t *)air_dev_mem_alloc(TILE_SIZE * sizeof(uint32_t));

  if (dram_ptr_1 == NULL || dram_ptr_2 == NULL || dram_ptr_3 == NULL ||
      dram_ptr_4 == NULL || dram_ptr_5 == NULL) {
    std::cout << "Couldn't allocate device memory" << std::endl;
    return -1;
  }

  for (int i = 0; i < TILE_SIZE; i++) {
    dram_ptr_1[i] = 1;
    dram_ptr_2[i] = 2;
    dram_ptr_3[i] = 3;
    dram_ptr_4[i] = 4;
    dram_ptr_5[i] = 5;
  }

  // Send the packet to write the tiles
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, /*herd_id*/ 0, /*col*/ 2, /*direction*/ 1,
                       /*channel*/ 0, /*burst_len*/ 4, /*space*/ 2,
                       /*type*/ 2, /*id*/ 2,
                       air_dev_mem_get_pa(dram_ptr_2) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(uint32_t), TILE_HEIGHT,
                       TILE_WIDTH * sizeof(uint32_t), 1, 0, 1, 0);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_b =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_b, /*herd_id*/ 0, /*col*/ 2, /*direction*/ 1,
                       /*channel*/ 0, /*burst_len*/ 4, /*space*/ 2,
                       /*type*/ 3, /*id*/ 3,
                       air_dev_mem_get_pa(dram_ptr_3) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(uint32_t), TILE_HEIGHT,
                       TILE_WIDTH * sizeof(uint32_t), 1, 0, 1, 0);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_c, /*herd_id*/ 0, /*col*/ 2, /*direction*/ 1,
                       /*channel*/ 0, /*burst_len*/ 4, /*space*/ 2,
                       /*type*/ 4, /*id*/ 4,
                       air_dev_mem_get_pa(dram_ptr_4) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(uint32_t), TILE_HEIGHT,
                       TILE_WIDTH * sizeof(uint32_t), 1, 0, 1, 0);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_d =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_d, /*herd_id*/ 0, /*col*/ 2, /*direction*/ 1,
                       /*channel*/ 0, /*burst_len*/ 4, /*space*/ 2,
                       /*type*/ 5, /*id*/ 5,
                       air_dev_mem_get_pa(dram_ptr_5) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(uint32_t), TILE_HEIGHT,
                       TILE_WIDTH * sizeof(uint32_t), 1, 0, 1, 0);

  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_d);
  printf("finished broadcast\n");

  printf("7,2 a : ");
  for (int i = 0; i < TILE_WIDTH; i++) {
    printf("%x ", mlir_aie_read_buffer_buf0(xaie, i));
  }
  printf("\n7,2 b : ");
  for (int i = 0; i < TILE_WIDTH; i++) {
    printf("%x ", mlir_aie_read_buffer_buf1(xaie, i));
  }
  printf("\n8,2 a : ");
  for (int i = 0; i < TILE_WIDTH; i++) {
    printf("%x ", mlir_aie_read_buffer_buf3(xaie, i));
  }
  printf("\n8,2 b : ");
  for (int i = 0; i < TILE_WIDTH; i++) {
    printf("%x ", mlir_aie_read_buffer_buf4(xaie, i));
  }
  printf("\n7,3 a : ");
  for (int i = 0; i < TILE_WIDTH; i++) {
    printf("%x ", mlir_aie_read_buffer_buf6(xaie, i));
  }
  printf("\n7,3 b : ");
  for (int i = 0; i < TILE_WIDTH; i++) {
    printf("%x ", mlir_aie_read_buffer_buf7(xaie, i));
  }
  printf("\n8,3 a : ");
  for (int i = 0; i < TILE_WIDTH; i++) {
    printf("%x ", mlir_aie_read_buffer_buf9(xaie, i));
  }
  printf("\n8,3 b : ");
  for (int i = 0; i < TILE_WIDTH; i++) {
    printf("%x ", mlir_aie_read_buffer_buf10(xaie, i));
  }
  printf("\n");

  // c

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_e =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_e, /*herd_id*/ 0, /*col*/ 3, /*direction*/ 1,
                       /*channel*/ 0, /*burst_len*/ 4, /*space*/ 2,
                       /*type*/ 0, /*id*/ 0,
                       air_dev_mem_get_pa(dram_ptr_1) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(uint32_t), TILE_HEIGHT,
                       TILE_WIDTH * sizeof(uint32_t), 1, 0, 1, 0);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_f =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_f, /*herd_id*/ 0, /*col*/ 3, /*direction*/ 1,
                       /*channel*/ 1, /*burst_len*/ 4, /*space*/ 2,
                       /*type*/ 0, /*id*/ 0,
                       air_dev_mem_get_pa(dram_ptr_1) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(uint32_t), TILE_HEIGHT,
                       TILE_WIDTH * sizeof(uint32_t), 1, 0, 1, 0);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_g =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_g, /*herd_id*/ 0, /*col*/ 6, /*direction*/ 1,
                       /*channel*/ 0, /*burst_len*/ 4, /*space*/ 2,
                       /*type*/ 0, /*id*/ 0,
                       air_dev_mem_get_pa(dram_ptr_1) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(uint32_t), TILE_HEIGHT,
                       TILE_WIDTH * sizeof(uint32_t), 1, 0, 1, 0);

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt_h =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_h, /*herd_id*/ 0, /*col*/ 6, /*direction*/ 1,
                       /*channel*/ 1, /*burst_len*/ 4, /*space*/ 2,
                       /*type*/ 0, /*id*/ 0,
                       air_dev_mem_get_pa(dram_ptr_1) /*AIR_BBUFF_BASE*/,
                       TILE_WIDTH * sizeof(uint32_t), TILE_HEIGHT,
                       TILE_WIDTH * sizeof(uint32_t), 1, 0, 1, 0);

  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt_h);
  printf("finished copies\n");

  sleep(1);

  uint32_t errs = 0;
  for (int i = 0; i < TILE_WIDTH; i++) {
    if (mlir_aie_read_buffer_buf2(xaie, i) != 9)
      errs++;
    if (mlir_aie_read_buffer_buf5(xaie, i) != 13)
      errs++;
    if (mlir_aie_read_buffer_buf8(xaie, i) != 11)
      errs++;
    if (mlir_aie_read_buffer_buf11(xaie, i) != 16)
      errs++;
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
