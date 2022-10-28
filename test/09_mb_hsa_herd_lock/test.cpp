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
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <vector>

#include "air.hpp"
#include "test_library.h"

#include "aie_inc.cpp"

int main(int argc, char *argv[])
{
  auto col = 7;
  auto row = 2;

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

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();
  if (xaie == NULL) {
    std::cout << "Error initializing libxaie" << std::endl;
    return -1;
  }

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;

  auto herd_id = 0;
  auto num_rows = 4;
  auto num_cols = 2;
  auto lock_id = 0;

  // herd_setup packet
  // Set up a 2x4 herd starting 7,2
  dispatch_packet_t *herd_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, num_cols, row, num_rows);
  //air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // reserve another packet in the queue
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;

  dispatch_packet_t *dev_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_device_init(dev_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(queues[0], wr_idx, dev_pkt);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  // mlir_aie_start_cores(xaie);

  // reserve another packet in the queue
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;

  // lock packet
  dispatch_packet_t *lock_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_aie_lock_range(lock_pkt, herd_id, lock_id, /*acq_rel*/0,
                            /*value*/0, 0, num_cols, 0, num_rows);
  air_queue_dispatch_and_wait(queues[0], wr_idx, lock_pkt);

  u32 errors = 0;
  for (int c = col; c < col + num_cols; c++)
    for (int r = row; r < row + num_rows; r++) {
      u32 locks = mlir_aie_read32(xaie, mlir_aie_get_tile_addr(xaie, c, r) + 0x0001EF00);
      if (locks != 0x1)
        errors++;
    }

  if (errors) {
    printf("%d errors\n", errors);
    for (int c = col; c < col + num_cols; c++)
      for (int r = row; r < row + num_rows; r++) {
        u32 locks = mlir_aie_read32(xaie, mlir_aie_get_tile_addr(xaie, c, r) + 0x0001EF00);
        printf("C[%d][%d] %08X\n", c, r, locks);
      }
  }
  else {
    // Release the herd locks!
    wr_idx = queue_add_write_index(queues[0], 1);
    packet_id = wr_idx % queues[0]->size;

    dispatch_packet_t *release_pkt =
        (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
    air_packet_aie_lock_range(release_pkt, herd_id, lock_id, /*acq_rel*/1,
                            /*value*/1, 0, num_cols, 0, num_rows);
    air_queue_dispatch_and_wait(queues[0], wr_idx, release_pkt);

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

  if (errors == 0) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }
}
