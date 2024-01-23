//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <vector>

#include "air.hpp"
#include "test_library.h"

#include "aie_inc.cpp"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define XAIE_NUM_COLS 10

int main(int argc, char *argv[]) {
  auto col = 7;
  auto row = 2;

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<hsa_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  uint32_t aie_max_queue_size = 0;
  hsa_agent_get_info(agents[0], HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                     &aie_max_queue_size);

  std::cout << "Max AIE queue size: " << aie_max_queue_size << std::endl;

  // Creating a queue
  std::vector<hsa_queue_t *> queues;
  hsa_queue_t *q = NULL;
  auto queue_create_status =
      hsa_queue_create(agents[0], aie_max_queue_size, HSA_QUEUE_TYPE_SINGLE,
                       nullptr, nullptr, 0, 0, &q);

  if (queue_create_status != HSA_STATUS_SUCCESS) {
    std::cout << "hsa_queue_create failed" << std::endl;
  }

  // Adding to our vector of queues
  queues.push_back(q);
  assert(queues.size() > 0 && "No queues were sucesfully created!");

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
  if (xaie == NULL) {
    std::cout << "Error initializing libxaie" << std::endl;
    return -1;
  }

  auto herd_id = 0;
  auto num_rows = 4;
  auto num_cols = 2;
  auto lock_id = 0;

  // Initialize the device and the segment
  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t segment_pkt;
  air_packet_segment_init(&segment_pkt, 0, col, num_cols, row, num_rows);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &segment_pkt);

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t shim_pkt;
  air_packet_device_init(&shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &shim_pkt);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  // mlir_aie_start_cores(xaie);

  // reserve another packet in the queue and create a lock_range
  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t lock_pkt;
  air_packet_aie_lock_range(&lock_pkt, herd_id, lock_id, 0, 0, 0, num_cols, 0,
                            num_rows);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &lock_pkt);

  u32 errors = 0;
  for (int c = col; c < col + num_cols; c++)
    for (int r = row; r < row + num_rows; r++) {
      u32 locks = mlir_aie_read32(xaie, mlir_aie_get_tile_addr(xaie, c, r) +
                                            0x0001EF00);
      if (locks != 0x1)
        errors++;
    }

  if (errors) {
    printf("%d errors\n", errors);
    for (int c = col; c < col + num_cols; c++)
      for (int r = row; r < row + num_rows; r++) {
        u32 locks = mlir_aie_read32(xaie, mlir_aie_get_tile_addr(xaie, c, r) +
                                              0x0001EF00);
        printf("C[%d][%d] %08X\n", c, r, locks);
      }
  } else {
    // Release the herd locks!
    wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
    packet_id = wr_idx % queues[0]->size;
    hsa_agent_dispatch_packet_t release_pkt;
    air_packet_aie_lock_range(&release_pkt, herd_id, lock_id, 1, 1, 0, num_cols,
                              0, num_rows);
    air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                                &release_pkt);

    for (int c = col; c < col + num_cols; c++)
      for (int r = row; r < row + num_rows; r++) {
        u32 locks = mlir_aie_read32(xaie, mlir_aie_get_tile_addr(xaie, c, r) +
                                              0x0001EF00);
        if (locks != 0x2)
          errors++;
      }

    if (errors) {
      for (int c = col; c < col + num_cols; c++)
        for (int r = row; r < row + num_rows; r++) {
          u32 locks = mlir_aie_read32(xaie, mlir_aie_get_tile_addr(xaie, c, r) +
                                                0x0001EF00);
          printf("C[%d][%d] %08X\n", col, row, locks);
        }
    }
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);

  // Shutdown AIR and HSA
  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    errors++;
  }

  if (errors == 0) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail.\n");
    return -1;
  }
}
