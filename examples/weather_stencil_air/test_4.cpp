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
  uint64_t shim_one_col = 2;
  uint64_t shim_two_col = 3;

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
  air_packet_segment_init(&segment_pkt, 0, shim_one_col, 8, row, 8);
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
  if (ddr_ptr_in_0 == NULL) {
    printf("Failed to allocate ddr_ptr_in_0\n");
    return 1;
  }

  uint32_t *ddr_ptr_in_1 =
      (uint32_t *)air_malloc(DMA_COUNT_IN * sizeof(uint32_t));
  if (ddr_ptr_in_1 == NULL) {
    printf("Failed to allocate ddr_ptr_in_1\n");
    return 1;
  }

  uint32_t *ddr_ptr_in_2 =
      (uint32_t *)air_malloc(DMA_COUNT_IN * sizeof(uint32_t));
  if (ddr_ptr_in_2 == NULL) {
    printf("Failed to allocate ddr_ptr_in_2\n");
    return 1;
  }

  uint32_t *ddr_ptr_in_3 =
      (uint32_t *)air_malloc(DMA_COUNT_IN * sizeof(uint32_t));
  if (ddr_ptr_in_3 == NULL) {
    printf("Failed to allocate ddr_ptr_in_3\n");
    return 1;
  }

  uint32_t *ddr_ptr_out_0 =
      (uint32_t *)air_malloc(DMA_COUNT_OUT * sizeof(uint32_t));
  if (ddr_ptr_out_0 == NULL) {
    printf("Failed to allocate ddr_ptr_out_0\n");
    return 1;
  }

  uint32_t *ddr_ptr_out_1 =
      (uint32_t *)air_malloc(DMA_COUNT_OUT * sizeof(uint32_t));
  if (ddr_ptr_out_1 == NULL) {
    printf("Failed to allocate ddr_ptr_out_1\n");
    return 1;
  }

  uint32_t *ddr_ptr_out_2 =
      (uint32_t *)air_malloc(DMA_COUNT_OUT * sizeof(uint32_t));
  if (ddr_ptr_out_2 == NULL) {
    printf("Failed to allocate ddr_ptr_out_2\n");
    return 1;
  }

  uint32_t *ddr_ptr_out_3 =
      (uint32_t *)air_malloc(DMA_COUNT_OUT * sizeof(uint32_t));
  if (ddr_ptr_out_3 == NULL) {
    printf("Failed to allocate ddr_ptr_out_3\n");
    return 1;
  }

  // initialize the external buffers
  for (int i = 0; i < DMA_COUNT_IN; i++) {
    *(ddr_ptr_in_0 + i) = i; // input
    *(ddr_ptr_in_1 + i) = i; // input
    *(ddr_ptr_in_2 + i) = i; // input
    *(ddr_ptr_in_3 + i) = i; // input
  }

  for (int i = 0; i < DMA_COUNT_OUT; i++) {
    *(ddr_ptr_out_0 + i) = 0; // input
    *(ddr_ptr_out_1 + i) = 0; // input
    *(ddr_ptr_out_2 + i) = 0; // input
    *(ddr_ptr_out_3 + i) = 0; // input
  }

  printf("Finish configure\n");

  //////////////////////////////////////// B Block 0
  //
  // send the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt;
  air_packet_nd_memcpy(&pkt, 0, shim_one_col, 1, 0, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_in_0),
                       DMA_COUNT_IN * sizeof(float), 1, 0, 1, 0, 1, 0);
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt);

  //
  // read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt2;
  air_packet_nd_memcpy(&pkt2, 0, shim_one_col, 0, 0, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_out_0),
                       DMA_COUNT_OUT * sizeof(float), 1, 0, 1, 0, 1, 0);
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt2.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt2);

  //////////////////////////////////////// B Block 1
  //
  // send the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt3;
  air_packet_nd_memcpy(&pkt3, 0, shim_one_col, 1, 1, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_in_1),
                       DMA_COUNT_IN * sizeof(float), 1, 0, 1, 0, 1, 0);
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt3.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt3);

  //
  // read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt4;
  air_packet_nd_memcpy(&pkt4, 0, shim_one_col, 0, 1, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_out_1),
                       DMA_COUNT_OUT * sizeof(float), 1, 0, 1, 0, 1, 0);
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt4.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt4);

  //////////////////////////////////////// B Block 2
  //
  // send the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt5;
  air_packet_nd_memcpy(&pkt5, 0, shim_two_col, 1, 0, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_in_2),
                       DMA_COUNT_IN * sizeof(float), 1, 0, 1, 0, 1, 0);
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt5.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt5);

  //
  // read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt6;
  air_packet_nd_memcpy(&pkt6, 0, shim_two_col, 0, 0, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_out_2),
                       DMA_COUNT_OUT * sizeof(float), 1, 0, 1, 0, 1, 0);
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt6.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt6);

  //////////////////////////////////////// B Block 3
  //
  // send the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt7;
  air_packet_nd_memcpy(&pkt7, 0, shim_two_col, 1, 1, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_in_3),
                       DMA_COUNT_IN * sizeof(float), 1, 0, 1, 0, 1, 0);
  hsa_amd_signal_create_on_agent(1, 0, nullptr, &agents[0], 0,
                                 &pkt7.completion_signal);
  air_write_pkt<hsa_agent_dispatch_packet_t>(queues[0], packet_id, &pkt7);

  //
  // read the data
  //

  wr_idx = hsa_queue_add_write_index_relaxed(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  hsa_agent_dispatch_packet_t pkt8;
  air_packet_nd_memcpy(&pkt8, 0, shim_two_col, 0, 1, 4, 2,
                       reinterpret_cast<uint64_t>(ddr_ptr_out_3),
                       DMA_COUNT_OUT * sizeof(float), 1, 0, 1, 0, 1, 0);

  // Dispatch the packets and afterwards destroy the completion signals
  air_queue_dispatch_and_wait(&agents[0], queues[0], packet_id, wr_idx, &pkt8);
  hsa_signal_destroy(pkt.completion_signal);
  hsa_signal_destroy(pkt2.completion_signal);
  hsa_signal_destroy(pkt3.completion_signal);
  hsa_signal_destroy(pkt4.completion_signal);
  hsa_signal_destroy(pkt5.completion_signal);
  hsa_signal_destroy(pkt6.completion_signal);
  hsa_signal_destroy(pkt7.completion_signal);

  for (int i = 0; i < 512; i++) {

    if (ddr_ptr_out_0[i] != 514 + i) {
      printf("[ERROR] 0x%x != 0x%x\n", ddr_ptr_out_0[i], 514 + i);
      errors++;
    }

    if (ddr_ptr_out_0[i] != ddr_ptr_out_1[i]) {
      printf("[ERROR] ddr_ptr_out_0[%d] (%d) != ddr_ptr_out_1[%d (%d)]\n", i,
             ddr_ptr_out_0[i], i, ddr_ptr_out_1[i]);
      errors++;
    }

    if (ddr_ptr_out_0[i] != ddr_ptr_out_2[i]) {
      printf("[ERROR] ddr_ptr_out_0[%d] (%d) != ddr_ptr_out_2[%d (%d)]\n", i,
             ddr_ptr_out_0[i], i, ddr_ptr_out_2[i]);
      errors++;
    }

    if (ddr_ptr_out_0[i] != ddr_ptr_out_3[i]) {
      printf("[ERROR] ddr_ptr_out_0[%d] (%d) != ddr_ptr_out_3[%d (%d)]\n", i,
             ddr_ptr_out_0[i], i, ddr_ptr_out_3[i]);
      errors++;
    }

    // printf("Location %d:  %d\n", i, ddr_ptr_out_0[i]);
    // printf("Location %d:  %d\n", i, ddr_ptr_out_1[i]);
    // printf("Location %d:  %d\n", i, ddr_ptr_out_2[i]);
    // printf("Location %d:  %d\n", i, ddr_ptr_out_3[i]);
  }

  // destroying the queue
  hsa_queue_destroy(queues[0]);
  air_free(ddr_ptr_in_0);
  air_free(ddr_ptr_out_0);
  air_free(ddr_ptr_in_1);
  air_free(ddr_ptr_out_1);
  air_free(ddr_ptr_in_2);
  air_free(ddr_ptr_out_2);
  air_free(ddr_ptr_in_3);
  air_free(ddr_ptr_out_3);

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
