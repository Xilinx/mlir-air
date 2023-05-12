//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <vector>

#include "air.hpp"

// Defined in air_queue.h
// typedef enum {
//  AIR_AGENT_INFO_NAME = 0,        // NUL-terminated char[8]
//  AIR_AGENT_INFO_VENDOR_NAME = 1, // NUL-terminated char[8]
//  AIR_AGENT_INFO_CONTROLLER_ID = 2,
//  AIR_AGENT_INFO_FIRMWARE_VER = 3,
//  AIR_AGENT_INFO_NUM_REGIONS = 4,
//  AIR_AGENT_INFO_HERD_SIZE = 5,
//  AIR_AGENT_INFO_HERD_ROWS = 6,
//  AIR_AGENT_INFO_HERD_COLS = 7,
//  AIR_AGENT_INFO_TILE_DATA_MEM_SIZE = 8,
//  AIR_AGENT_INFO_TILE_PROG_MEM_SIZE = 9,
//  AIR_AGENT_INFO_L2_MEM_SIZE = 10 // Per region
//} air_agent_info_t;

int main(int argc, char *argv[]) {

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
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

  // Sending a read32write32 packet to the ARM
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_rw32_init(pkt, 0, 0x72004, 0xDEADBEEF /* Doesn't matter because read */);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt);
  printf("We have returned a value of 0x%lx\n", pkt->return_address);

  std::cout << std::endl << "PASS!" << std::endl;
  return 0;
}
