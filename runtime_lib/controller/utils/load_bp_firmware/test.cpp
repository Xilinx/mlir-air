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
#include <stdio.h>

#include <sstream>
#include <iomanip>

#include "air.hpp"

using namespace std;

int main(int argc, char *argv[]) {

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }


  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(agents);
  assert(ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  // Getting the binary to load 
  std::string file_name = "main.mem";
  if(argc > 1) {
    file_name = argv[1];
  } 
  FILE *mem_fd = fopen(file_name.c_str(), "rb");
  if(mem_fd == NULL) {
    printf("[ERROR] Cannot find file %s\n", file_name.c_str());
    return -1;
  }
  // Need to get the size of the file
  fseek(mem_fd, 0, SEEK_END);
  uint32_t file_size = ftell(mem_fd);
  fseek(mem_fd, 0, SEEK_SET); // Have to reset the ptr

  // Need to get the number of lines of the file
  uint32_t file_num_lines = 0;
  char * num_lines_line = NULL;
  size_t num_lines_len = 0;
  ssize_t num_lines_read;
  while((num_lines_read = getline(&num_lines_line, &num_lines_len, mem_fd)) != -1) {
    file_num_lines++;
  }
  
  printf("Loading elf from %s of size %d and %d lines into all BPs\n", file_name.c_str(), file_size, file_num_lines);

  // Use mmap to treat the file like an array
  void *elf_host_mem = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fileno(mem_fd), 0);

  // Allocating device memory 
  if (air_init_dev_mem_allocator(0x100000)) {
    std::cout << "Error creating device memory allocator" << std::endl;
    return -1;
  }

  void *elf_dev_mem = air_dev_mem_alloc(file_size);

  if (elf_dev_mem == nullptr) {
    std::cout << "Could not allocate necessary device memory" << std::endl;
    return -1;
  }
  printf("Allocating firmware at device PA: 0x%lx\n", air_dev_mem_get_pa(elf_dev_mem));

  // Copying the contents of the elf from the host to the device
  memcpy(elf_dev_mem, elf_host_mem, file_size);

  // Just want to create a single queue for the ARM
  queue_t *dev_ctrl_queue = nullptr;
  ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &dev_ctrl_queue, agents[0].handle);
  assert(ret == 0 && "failed to create queue!");

  // Program the firmware of the BPs for every agent other than the ARM
  uint64_t wr_idx = queue_add_write_index(dev_ctrl_queue, 1);
  uint64_t packet_id = wr_idx % dev_ctrl_queue->size;
  dispatch_packet_t *pkt = (dispatch_packet_t *)(dev_ctrl_queue->base_address_vaddr) + packet_id;
  printf("Creating program_firmware packet\n");
  air_program_firmware(pkt, air_dev_mem_get_pa(elf_dev_mem), file_num_lines);
  
  // Spinning on the signal here so we don't have the runtime complaining about timeouts
  printf("Sending packet\n");
  air_queue_dispatch(dev_ctrl_queue, wr_idx, pkt);
  printf("Waiting on completion...\n");
  while (signal_wait_acquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0);

  air_dev_mem_allocator_free();

  std::cout << std::endl << "PASS!" << std::endl;
  return 0;
}
