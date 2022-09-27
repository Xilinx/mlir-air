//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
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

#include <assert.h>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <vector>

#include "acdc_queue.h"
#include "air_host.h"
#include "hsa_defs.h"

// Defined in acdc_queue.h
//typedef enum {
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
  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(&agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
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
 
  uint64_t data = -1;
  char vend[8];
  for (auto q : queues) {
    std::cout << std::endl << "Requesting attribute: AIR_AGENT_INFO_CONTROLLER_ID... ";
    air_get_agent_info(q, AIR_AGENT_INFO_CONTROLLER_ID, &data);
    std::cout << "Agent ID is: " << data << std::endl;

    std::cout << "Requesting attribute: AIR_AGENT_INFO_VENDOR_NAME... ";
    air_get_agent_info(q, AIR_AGENT_INFO_VENDOR_NAME, vend);
    std::cout << "Vendor is: " << vend << std::endl;

    std::cout << "Requesting attribute: AIR_AGENT_INFO_L2_MEM_SIZE... ";
    air_get_agent_info(q, AIR_AGENT_INFO_L2_MEM_SIZE, &data);
    std::cout << "L2 size is: " << std::dec << data << "B" << std::endl;
  }

  std::cout << std::endl << "PASS!" << std::endl;
  return 0;
}
