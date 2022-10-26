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

#include "air_host.h"

#define SCRATCH_AREA 8

#include "aie_inc.cpp"

int
main(int argc, char *argv[])
{
  auto col = 7;
  auto row = 2;

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

  // create the queue
  /*queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
  AIR_VCK190_SHMEM_BASE); assert(ret == 0 && "failed to create queue!");*/

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;

  // herd_setup packet
  dispatch_packet_t *herd_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 1);
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

  // mlir_aie_print_tile_status(xaie, col, row);

  // We first write an ascending pattern into the area the AIE will write into
  for (int i=0; i<SCRATCH_AREA; i++) {
    uint32_t d = i+1;
    mlir_aie_write_buffer_buffer(xaie, i, d);
  }

  uint32_t herd_id = 0;
  uint32_t lock_id = 0;

  // We wrote data, so lets tell the MicroBlaze to toggle the job lock 0
  // reserve another packet in the queue
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;

  // lock packet
  dispatch_packet_t *lock_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_aie_lock(lock_pkt, herd_id, lock_id, /*acq_rel*/1, /*value*/1, 0, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, lock_pkt);

  auto count = 0;
  while (!mlir_aie_acquire_lock(xaie, col, 2, 0, 0, 1000)) {
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  int errors = 0;
  mlir_aie_check("Check Result 0:", mlir_aie_read_buffer_buffer(xaie, 0), 0xdeadbeef,errors);
  mlir_aie_check("Check Result 1:", mlir_aie_read_buffer_buffer(xaie, 1), 0xcafecafe,errors);
  mlir_aie_check("Check Result 2:", mlir_aie_read_buffer_buffer(xaie, 2), 0x000decaf,errors);
  mlir_aie_check("Check Result 3:", mlir_aie_read_buffer_buffer(xaie, 3), 0x5a1ad000,errors);

  for (int i=4; i<SCRATCH_AREA; i++)
    mlir_aie_check("Check Result:", mlir_aie_read_buffer_buffer(xaie, i), i+1,errors);

  if (!errors) {
    printf("PASS!\n");
    return 0;
      }
  else {
    printf("fail %d/%d.\n", (SCRATCH_AREA-errors), SCRATCH_AREA);
    return -1;
      }
}
