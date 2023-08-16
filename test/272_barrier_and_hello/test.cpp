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
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "air.hpp"

int main(int argc, char *argv[]) {

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.size() < 2) {
    std::cout << "WARNING: Test is unsuported with < 2 queues." << std::endl;
    return 0;
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

  // Put 5 hello packets in agent 0's queue
  dispatch_packet_t *p;
  uint32_t db = 0;
  std::vector<uint64_t> signals;
  for (int i = 0; i < 5; i++) {
    uint64_t wr_idx = queue_add_write_index(queues[0], 1);
    uint64_t packet_id = wr_idx % queues[0]->size;

    dispatch_packet_t *pkt =
        (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
    air_packet_hello(pkt, 0xacdc0000LL + i);
    signal_create(1, 0, NULL, (signal_t *)&pkt->completion_signal);

    uint64_t signal_offset = offsetof(dispatch_packet_t, completion_signal);
    uint64_t s = queue_paddr_from_index(
        queues[0], (packet_id) * sizeof(dispatch_packet_t) + signal_offset);
    signals.push_back(s);
    // signals.push_back((uint64_t)&pkt->completion_signal);
    p = pkt;
    db = wr_idx;
  }

  // Put a hello packet in agent 1's queue
  uint64_t wr_idx = queue_add_write_index(queues[1], 1);
  uint64_t packet_id = wr_idx % queues[1]->size;

  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(queues[1]->base_address_vaddr) + packet_id;
  air_packet_hello(pkt, 0xfeed);

  // Put a barrier AND packet in agent 1's queue
  wr_idx = queue_add_write_index(queues[1], 1);
  packet_id = wr_idx % queues[1]->size;

  barrier_and_packet_t *barrier_pkt =
      (barrier_and_packet_t *)(queues[1]->base_address_vaddr) + packet_id;
  air_packet_barrier_and(barrier_pkt, signals[0], signals[1], signals[2],
                         signals[3], signals[4]);
  signal_create(1, 0, NULL, (signal_t *)&barrier_pkt->completion_signal);

  // Put another hello packet in agent 1's queue
  wr_idx = queue_add_write_index(queues[1], 1);
  packet_id = wr_idx % queues[1]->size;

  pkt = (dispatch_packet_t *)(queues[1]->base_address_vaddr) + packet_id;
  air_packet_hello(pkt, 0xface);

  // Dispatch to start agent 1 
  // Second air_hello blocked by barrier_and waiting 
  // for agent 0 to finish
  std::cout << "Dispatch work to MB 1 -> "
            << "] air_hello ] barrier_and ] air_hello ]" << std::endl
            << std::endl;
  air_queue_dispatch(queues[1], wr_idx, pkt);

  std::cout << "Waiting to dispatch work to MB 0 ";
  fflush(stdout);
  for (int i = 0; i < 5; i++) {
    std::cout << " ... ";
    fflush(stdout);
    sleep(1);
  }
  std::cout << std::endl
            << "Dispatch work to MB 0 -> ] 5x air_hello packets ]"
            << std::endl;

  // Hit doorbell of agent 0's queue to start processing
  signal_create(0, 0, NULL, (signal_t *)&queues[0]->doorbell);
  signal_store_release((signal_t *)&queues[0]->doorbell, db);
  air_queue_wait(queues[0], p);

  air_queue_wait(queues[1], pkt);

  std::cout << std::endl << "PASS!" << std::endl;
  return 0;
}
