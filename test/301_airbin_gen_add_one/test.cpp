// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cstdint>  // uint64_t
#include <cstdio>   // printf
#include <cstdlib>  // atoi
#include <ctime>    // clock_gettime
#include <fcntl.h>  // open
#include <fstream>  // ifstream
#include <iomanip>  // setw, dec, hex
#include <iostream> // cout
#include <string>
#include <sys/mman.h>
#include <sys/time.h>
#include <vector>

#include "air.hpp"

int addone_driver(queue_t *q, int col, int row) {
  /////////////////////////////////////////////////////////////////////////////
  //////////////////////// Run Add One Application ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

#define DMA_COUNT 16

  // setup images in memory
  uint32_t *bb_ptr;
  uint64_t paddr = 0x2000;

  // Initializing the device memory allocator
  if (air_init_dev_mem_allocator(0x8000 /* dev_mem_size */,
                                 0 /* device_id (optional)*/)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  bb_ptr = (uint32_t *)air_dev_mem_alloc(0x100000);
  paddr = air_dev_mem_get_pa(bb_ptr);
  for (int i = 0; i < DMA_COUNT; i++) {
    bb_ptr[i] = i + 1;
    bb_ptr[DMA_COUNT + i] = 0xdeface;
  }

  //
  // send the data
  //

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt1 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt1, 0, col, 1, 0, 4, 2, paddr,
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt2 =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2,
                       paddr + (DMA_COUNT * sizeof(float)),
                       DMA_COUNT * sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt2);

  int errors = 0;

  for (uint32_t i = 0; i < DMA_COUNT; i++) {
    volatile uint32_t d = bb_ptr[DMA_COUNT + i];
    if (d != (i + 2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  air_dev_mem_allocator_free();

  if (!errors) {
    return 0;
  } else {
    return 1;
  }
}

int main(int argc, char **argv) {
  int col = 7;
  int row = 2;
  std::string airbin_name;

  if (argc > 3) {
    std::cout << "Usage: " << *argv << " [airbin file] [column number]"
              << std::endl;
    return 1;
  } else if (argc == 3) {
    if (auto column_input = atoi(argv[2]);
        column_input > 0 and column_input < UINT8_MAX) {
      col = column_input;
    } else {
      std::cout << "Error: " << argv[2] << " must be between 0 and "
                << UINT8_MAX << " inclusive" << std::endl;
      return 2;
    }
    airbin_name = argv[1];
  } else if (argc == 2) {
    airbin_name = argv[1];
  } else {
    airbin_name = "addone.airbin";
  }

  std::cout << "\nConfiguring herd in col " << col << "..." << std::endl;
  std::cout << "\nConfiguring file " << airbin_name << "..." << std::endl;

  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                                agent.handle);
    assert(ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  auto *q = queues[0];

  if (air_load_airbin(q, airbin_name.c_str(), col) != 0) {
    std::cout << "Error loading airbin" << std::endl;
    return 1;
  }

  std::cout << "\nDone configuring!" << std::endl << std::endl;

  int errors = 0;

  if (airbin_name == "addone.airbin") {
    errors = addone_driver(q, col, row);
  } else
    errors = 1;

  if (!errors) {
    std::cout << "PASS!" << std::endl;
    return 0;
  } else {
    std::cout << "fail." << std::endl;
    return -1;
  }
}
