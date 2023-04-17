// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xaiengine.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

// ERNIC includes 
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <signal.h>
#include <ctype.h>
#include <termios.h>
#include <sys/types.h>

#include "air.hpp"
#include "test_library.h"
#include "air_tensor.h"
#include "air_host.h"
#include "air_network.h"
#include "pcie-ernic.h"

#include "aie_inc.cpp"

// For debugging information want 
// to have an ID for the ERNIC we 
// are talking to
#define ERNIC_ID 1

// For AIE level tests that use RDMA need to pull the internal data 
// structures that keep track of remote and local buffers out of the runtime
extern std::map<void*, tensor_to_qp_map_entry*>tensor_to_qp_map;

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
void init_data_placement(std::map<std::string, std::string> &data_placement) {
  data_placement["src"] = std::string("host_1");
  data_placement["dst"] = std::string("host_0");
  data_placement["host0_barrier_tensor"] = std::string("host_0");
  data_placement["host1_barrier_tensor"] = std::string("host_1");
}


int
main(int argc, char *argv[])
{
  uint64_t row = 0;
  uint64_t col = 6;

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

  std::cout << "Found " << agents.size() << " agents" << std::endl;

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

  //
  // Set up a 1x3 herd starting 7,0
  //
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *herd_pkt =
      (dispatch_packet_t *)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
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

  #define DMA_COUNT 16
  #define BUFFER_COUNT (DMA_COUNT*2) // Want the buffer to be twice the size as the DMA

  for (int i=0; i<8; i++) {
    mlir_aie_write_buffer_ping_in(xaie, i, 0xabbaba00+i);
    mlir_aie_write_buffer_pong_in(xaie, i, 0xdeeded00+i);
    mlir_aie_write_buffer_ping_out(xaie, i, 0x12345670+i);
    mlir_aie_write_buffer_pong_out(xaie, i, 0x76543210+i);
  }

  // Initializing the represenations of all remote AIR
  // instances and all remotely accessible buffers 
  std::map<std::string, world_view_entry *> world_view;
  init_world_view(world_view);
  std::map<std::string, std::string> data_placement;
  init_data_placement(data_placement);

  // Exploring the world to see if there are remote AIR
  // instances. If so, it will initialize the ERNIC
  // and create QPs to communicate with each AIR instance
  hsa_status_t hsa_ret = air_set_hostname("host_0");
  hsa_ret = air_explore_world(ERNIC_ID, 0x00020000 /* dev mem offset */, 0x00080000 /* axil bar offset */, world_view, data_placement);

  // Registering the memory for the src and destination
  tensor_t<uint32_t, 1> src_tensor, dst_tensor, barrier_dummy_tensor;
  hsa_ret = air_ernic_mem_alloc("src", BUFFER_COUNT * sizeof(uint32_t), &src_tensor, true);
  hsa_ret = air_ernic_mem_alloc("dst", BUFFER_COUNT * sizeof(uint32_t), &dst_tensor, true);
  hsa_ret = air_ernic_mem_alloc("host0_barrier_tensor", RQE_SIZE, &barrier_dummy_tensor, false);

  // Clang doesn't like pointer arithmetic on void pointers so 
  // need to use this other pointer to read the local buffer
  // of the remote tensor
  uint32_t *src_tensor_local_buff = (uint32_t *)(tensor_to_qp_map[src_tensor.alloc]->local_buff->buff);

  for (int i=0; i<BUFFER_COUNT; i++) {
    src_tensor_local_buff[i] = 0xaaaaaaaa;
    dst_tensor.alloc[i] = 0xdeface;
  }

  mlir_aie_print_tile_status(xaie,col,2);
  mlir_aie_print_dma_status(xaie,col,2);

  //
  // Reading the remote buffer 
  //

  // Wait for the src buffer to be initialized before continuing
  air_barrier(&barrier_dummy_tensor, queues[0], 0);

  printf("Generating READ POST_WQE HSA packet\n");
  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt0 = (dispatch_packet_t*)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_post_rdma_wqe(pkt0,                                                 // HSA Packet
                          tensor_to_qp_map[src_tensor.alloc]->vaddr,            // Remote VADDR
                          tensor_to_qp_map[src_tensor.alloc]->local_buff->pa,   // Local PADDR
                          BUFFER_COUNT * sizeof(float),                         // Length
                          (uint8_t)OP_READ,                                     // op
                          (uint8_t)tensor_to_qp_map[src_tensor.alloc]->rkey,    // Key
                          (uint8_t)tensor_to_qp_map[src_tensor.alloc]->qp,      // QPID
                          (uint8_t)0);                                          // ERNIC select
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt0);

  // Reading the data in the READ RESPONSE
  printf("Data before performing kernel:\n");
  for(int i = 0; i < BUFFER_COUNT; i++) {
    printf("src[%d] = 0x%x\n", i, src_tensor_local_buff[i]);
    printf("dst[%d] = 0x%x\n", i, dst_tensor.alloc[i]);
  }

  //
  // send the data
  //

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt1 = (dispatch_packet_t*)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt1, 0, col, 1, 0, 4, 2, tensor_to_qp_map[src_tensor.alloc]->local_buff->pa, DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt1);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(queues[0], 1);
  packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *pkt2 = (dispatch_packet_t*)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2, tensor_to_qp_map[dst_tensor.alloc]->local_buff->pa, DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(queues[0], wr_idx, pkt2);

  mlir_aie_print_tile_status(xaie,col,2);
  mlir_aie_print_dma_status(xaie,col,2);

  int errors = 0;

  for (int i=0; i<8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out(xaie, i);
    if (d0+1 != d2) {
      printf("mismatch ping %x != %x\n", d0, d2);
      errors++;
    }
    if (d1+1 != d3) {
      printf("mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  // Wait for other processes to write to the destination
  air_barrier(&barrier_dummy_tensor, queues[0], 0);

  for (int i=0; i<BUFFER_COUNT; i++) {
    //uint32_t s = src_tensor.d[i];
    volatile uint32_t s = src_tensor_local_buff[i];
    uint32_t d = dst_tensor.alloc[i];

    printf("src[%d] = 0x%x\n", i, s);
    printf("dst[%d] = 0x%x\n", i, d);

    if (d != (s+1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, s);
    }
  }

  // Free the ernic resources
  air_ernic_free();

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, BUFFER_COUNT);
    return -1;
  }

}
