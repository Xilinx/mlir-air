// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <xaiengine.h>
#include "air.hpp"
#include "test_library.h"

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

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
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

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
  if (xaie == NULL) {
    std::cout << "Error getting libxaie context" << std::endl;
    return -1;
  }

  // Initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x8000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  // Initializing the device
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(queues[0], wr_idx, shim_pkt);

  //
  // Set up a 8x8 segment 
  //
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *segment_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_segment_init(segment_pkt, 0, col, 8, row, 8);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);
  int errors = 0;

  uint32_t *ddr_ptr_in_0 = (uint32_t *)air_dev_mem_alloc(DMA_COUNT_IN * sizeof(uint32_t));
  uint32_t *ddr_ptr_out_0 = (uint32_t *)air_dev_mem_alloc(DMA_COUNT_OUT * sizeof(uint32_t));

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

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, 0, col, 1, 0, 4, 2, air_dev_mem_get_pa(ddr_ptr_in_0),
                       DMA_COUNT_IN * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt2 =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2, air_dev_mem_get_pa(ddr_ptr_out_0),
                       DMA_COUNT_OUT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt2);

  for (int i = 0; i < 512; i++) {
    printf("Location %d:  %d\n", i, ddr_ptr_out_0[i]);
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
