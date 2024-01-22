// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
// SPDX-License-Identifier: MIT

#include "air.hpp"
#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <xaiengine.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)
#define MLIR_STACK_OFFSET 4096

#define TOTAL_B_BLOCK 1 // only 1
#define B_BLOCK_DEPTH 4 // set how many rows
#define HDIFF_COL 3     // columns
#define START_ROW 1
#define INPUT_ROWS 9
#define DMA_COUNT_IN 256 * INPUT_ROWS
#define DMA_COUNT_OUT 256 * 2 * B_BLOCK_DEPTH
#define XAIE_NUM_COLS 10

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {

  // Starting in the first DU of the VCK5000
  uint64_t row = 0;
  uint64_t col = 2;

  std::vector<hsa_queue_t *> queues;
  uint32_t aie_max_queue_size(0);

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<hsa_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  hsa_agent_get_info(agents[0], HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                     &aie_max_queue_size);

  std::cout << "Max AIE queue size: " << aie_max_queue_size << std::endl;

  hsa_queue_t *q = NULL;

  // Creating a queue
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
    std::cout << "Error getting libxaie context" << std::endl;
    return -1;
  }

  //
  // Setting up the device and the segment
  //
  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t shim_pkt;
  air_packet_device_init(&shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &shim_pkt);

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t segment_pkt;
  air_packet_segment_init(&segment_pkt, 0, col, 8, row, 8);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx,
                              &segment_pkt);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);
  int errors = 0;

  uint32_t *ddr_ptr_in_0 =
      (uint32_t *)air_malloc(DMA_COUNT_IN * sizeof(uint32_t));
  uint32_t *ddr_ptr_out_0 =
      (uint32_t *)air_malloc(DMA_COUNT_OUT * sizeof(uint32_t));

  // initialize the external buffers
  for (int i = 0; i < DMA_COUNT_IN; i++) {
    *(ddr_ptr_in_0 + i) = i; // input
  }

  for (int i = 0; i < DMA_COUNT_OUT; i++) {
    *(ddr_ptr_out_0 + i) = 0; // input
  }

  printf("Finish configure\n");

  //
  // send the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt;
  air_packet_nd_memcpy(&pkt, 0, col, 1, 0, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_in_0),
                       DMA_COUNT_IN * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt);

  //
  // read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt2;
  air_packet_nd_memcpy(&pkt2, 0, col, 0, 0, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_out_0),
                       DMA_COUNT_OUT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt2);

  for (int i = 0; i < 1024; i++) {
    if (ddr_ptr_out_0[i] != 514 + i) {
      printf("[ERROR] 0x%x != 0x%x\n", ddr_ptr_out_0[i], 514 + i);
      errors++;
    }
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);
  air_free(ddr_ptr_in_0);
  air_free(ddr_ptr_out_0);

  // Shutdown AIR and HSA
  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    errors++;
  }

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  printf("test done weather predicted = chocolate :D\n");

  return 0;
}
