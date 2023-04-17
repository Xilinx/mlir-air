//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>

#include <xaiengine.h>

// ERNIC includes 
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <signal.h>
#include <ctype.h>
#include <termios.h>
#include <sys/types.h>

// AIR includes
#include "air.hpp"
#include "test_library.h"
#include "air_tensor.h"
#include "air_host.h"
#include "air_network.h"
#include "hsa_defs.h"
#include "pcie-ernic.h"

//#include "aie_inc.cpp"

#define ERNIC_ID 1

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

namespace air::partitions::partition_0 {
void mlir_aie_write_buffer_scratch_0_0(aie_libxaie_ctx_t*, int, int32_t);
};
using namespace air::partitions::partition_0;

// Creating the world_view so we know other AIR instances
// Eventually this will be read from a serialied data 
// structure sent from the launching node
void init_world_view(std::map<std::string, world_view_entry *> &world_view) {
  world_view["host_0"] = (struct world_view_entry *)malloc(sizeof(world_view_entry));
  strcpy(world_view["host_0"]->ip, "610c6007");
  strcpy(world_view["host_0"]->mac, "000016C450560F2E");
  world_view["host_0"]->rank = 0;
  world_view["host_0"]->qps[1] = 2;
  world_view["host_1"] = (struct world_view_entry *)malloc(sizeof(world_view_entry));
  strcpy(world_view["host_1"]->ip, "f38590ba");
  strcpy(world_view["host_1"]->mac, "00002F7617DC5E9A");
  world_view["host_1"]->rank = 1;
  world_view["host_1"]->qps[0] = 2;
}

// Creating the data placement so we know where all 
// remotely accessble buffers are located
// Eventually this will be read from a serialied data 
// structure sent from the launching node
void init_data_placement_case_0(std::map<std::string, std::string> &data_placement) {
  data_placement["src"] = std::string("host_1");
  data_placement["dst"] = std::string("host_0");
}

void init_data_placement_case_1(std::map<std::string, std::string> &data_placement) {
  data_placement["src"] = std::string("host_0");
  data_placement["dst"] = std::string("host_1");
}

void init_data_placement_case_2(std::map<std::string, std::string> &data_placement) {
  data_placement["src"] = std::string("host_1");
  data_placement["dst"] = std::string("host_1");
}

void init_data_placement_case_3(std::map<std::string, std::string> &data_placement) {
  data_placement["src"] = std::string("host_0");
  data_placement["dst"] = std::string("host_0");
}


int
main(int argc, char *argv[])
{

  if(argc != 2) {
    printf("USAGE: ./herd.exe {mode}\n");
    printf("\t0: Remote src buffer\n");
    printf("\t1: Remote dst buffer\n");
    printf("\t2: Remote src and dst buffer\n");
    printf("\t3: Local src and dst buffer\n");
    return 1;
  }

  int driver_mode = atoi(argv[1]);
  if(driver_mode > 3) {
    printf("[ERROR] No driver_mode %d supported\n", driver_mode);
    return 1;
  }

  uint64_t row = 3;
  uint64_t col = 3;

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == 0 && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    auto create_queue_ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                           agent.handle);
    assert(create_queue_ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
  if (xaie == NULL) {
    std::cout << "Error getting libxaie context" << std::endl;
    return -1;
  }
  
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(queues[0], wr_idx, shim_pkt);

  for (int i=0; i<TILE_SIZE; i++)
    mlir_aie_write_buffer_scratch_0_0(xaie, i, 0xfadefade);

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr,queues[0]);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void *))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  // Setting the world view and data placement
  std::map<std::string, world_view_entry *> world_view;
  std::map<std::string, std::string> data_placement;
  init_world_view(world_view);
  if(driver_mode == 0) {
    init_data_placement_case_0(data_placement);
  }
  else if(driver_mode == 1) {
    init_data_placement_case_1(data_placement);
  }
  else if(driver_mode == 2) {
    init_data_placement_case_2(data_placement);
  }
  else if (driver_mode == 3) {
    init_data_placement_case_3(data_placement);
  }
  else {
    printf("[ERROR] Unrecognized driver mode provided: %d\n", driver_mode);
    return -1;
  }

  // Exploring the world to see if there are remote AIR
  // instances. If so, it will initialize the ERNIC
  // and create QPs to communicate with each AIR instance
  hsa_status_t hsa_ret = air_set_hostname("host_0");
  hsa_ret = air_explore_world(ERNIC_ID, 0x00020000 /* dev mem offset */, 0x00080000 /* axil bar offset */, world_view, data_placement);

  tensor_t<uint32_t,2> input;
  tensor_t<uint32_t,2> output;

  input.shape[1] = IMAGE_WIDTH;
  input.shape[0] = IMAGE_HEIGHT; 
  hsa_ret = air_ernic_mem_alloc("src", sizeof(uint32_t)*input.shape[0]*input.shape[1], &input, false);

  output.shape[1] = IMAGE_WIDTH;
  output.shape[0] = IMAGE_HEIGHT; 
  hsa_ret = air_ernic_mem_alloc("dst", sizeof(uint32_t)*output.shape[0]*output.shape[1], &output, false);

  for (int i=0; i<IMAGE_SIZE; i++) {
    if(driver_mode == 1 || driver_mode == 3) {
      input.data[i]  = i+0x1000; // Input is now remote so will not write to it here
    }
    if(driver_mode == 0 || driver_mode == 3) {
      output.data[i] = 0x0defaced;
    }
  }

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  int errors = 0;

  // Now look at the image, should have the bottom left filled in
  if(driver_mode == 0 || driver_mode == 3) {
    printf("Checking output buffer because it is local\n");
    for (int i=0;i<IMAGE_SIZE;i++) {
      u32 rb = output.data[i];

      u32 row = i / IMAGE_WIDTH;
      u32 col = i % IMAGE_WIDTH;

      if ((row >= TILE_HEIGHT) && (col < TILE_WIDTH)) {
        if (!(rb == 0x1000+i)) {
          printf("IM %d [%d, %d] should be %08X, is %08X\n", i, col, row, i+0x1000, rb);
          errors++;
        }
      }
      else {
        if (rb != 0x00defaced) {
          printf("IM %d [%d, %d] should be 0xdefaced, is %08X\n", i, col, row, rb);
          errors++;
        }
      }
    }

    if (!errors) {
      printf("PASS!\n");
    }
    else {
      printf("fail %d/%d.\n", (TILE_SIZE+IMAGE_SIZE-errors), TILE_SIZE+IMAGE_SIZE);
    }  
  }

  // Free the ernic resources
  air_ernic_free();

  if (!errors) {
    return 0;
  }
  else {
    return -1;
  }

}
