// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xaiengine.h>

#include "air_host.h"
#include "acdc_queue.h"
#include "aie_inc.cpp"

int
main(int argc, char *argv[])
{
  bool use_barrier = true;
  
  int col = 7;
  int row = 2;
  int col2 = 34;

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  uint32_t *bram_ptr;

  #define BRAM_ADDR 0x4000+AIR_VCK190_SHMEM_BASE
  #define DMA_COUNT 16

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<4*DMA_COUNT; i++) {
      bram_ptr[i] = i+1;
      bram_ptr[4*DMA_COUNT+i] = 0xdeface;
      bram_ptr[8*DMA_COUNT+i] = 0xcafe00;
    }
  }

  for (int i=0; i<8; i++) {
    mlir_aie_write_buffer_t72_ping_in(xaie, i, 0xabbaba10+i);
    mlir_aie_write_buffer_t72_pong_in(xaie, i, 0xdeeded10+i);
    mlir_aie_write_buffer_t72_ping_out(xaie, i, 0x12345610+i);
    mlir_aie_write_buffer_t72_pong_out(xaie, i, 0x76543210+i);
    mlir_aie_write_buffer_t73_ping_in(xaie, i, 0xabbaba10+i);
    mlir_aie_write_buffer_t73_pong_in(xaie, i, 0xdeeded10+i);
    mlir_aie_write_buffer_t73_ping_out(xaie, i, 0x12345610+i);
    mlir_aie_write_buffer_t73_pong_out(xaie, i, 0x76543210+i);
    mlir_aie_write_buffer_t82_ping_in(xaie, i, 0xabbaba10+i);
    mlir_aie_write_buffer_t82_pong_in(xaie, i, 0xdeeded10+i);
    mlir_aie_write_buffer_t82_ping_out(xaie, i, 0x12345610+i);
    mlir_aie_write_buffer_t82_pong_out(xaie, i, 0x76543210+i);
    mlir_aie_write_buffer_t83_ping_in(xaie, i, 0xabbaba10+i);
    mlir_aie_write_buffer_t83_pong_in(xaie, i, 0xdeeded10+i);
    mlir_aie_write_buffer_t83_ping_out(xaie, i, 0x12345610+i);
    mlir_aie_write_buffer_t83_pong_out(xaie, i, 0x76543210+i);
    mlir_aie_write_buffer_ping_in2(xaie, i, 0xabbaba20+i);
    mlir_aie_write_buffer_pong_in2(xaie, i, 0xdeeded20+i);
    mlir_aie_write_buffer_ping_out2(xaie, i, 0x12345620+i);
    mlir_aie_write_buffer_pong_out2(xaie, i, 0x76543220+i);
  }

  std::vector<air_agent_t> agents;
  auto ret = air_get_agents(&agents);
  assert(ret == 0 && "failed to get agents!");

  if (agents.size() < 2) {
    std::cout << "fail." << std::endl;
    return -1;
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

  assert(queues.size() > 1 && "failed to create at least 2 queues!");

  //
  // Set up a 2x2 herd starting 7,2
  //
  uint64_t wr_idx = queue_add_write_index(queues[0], 1);
  uint64_t packet_id = wr_idx % queues[0]->size;
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(queues[0]->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 2, row, 2);
  //
  // Set up a 1x1 herd starting 34,2
  //
  uint64_t wr_idx2 = queue_add_write_index(queues[1], 1);
  uint64_t packet_id2 = wr_idx2 % queues[1]->size;
  dispatch_packet_t *herd_pkt2 = (dispatch_packet_t*)(queues[1]->base_address_vaddr) + packet_id2;
  air_packet_herd_init(herd_pkt2, 1, col2, 1, row, 1);

  dispatch_packet_t *p;
  std::vector<uint64_t> signals;
  uint64_t signal_offset = offsetof(dispatch_packet_t, completion_signal);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      //
      // send the data
      //

      wr_idx = queue_add_write_index(queues[0], 1);
      packet_id = wr_idx % queues[0]->size;
      dispatch_packet_t *pkt1 = (dispatch_packet_t*)(queues[0]->base_address_vaddr) + packet_id;
      air_packet_nd_memcpy(pkt1, 0, 2+i, 1, 0+j, 4, 2, 
                           BRAM_ADDR+((2*i+j)*DMA_COUNT*sizeof(float)), 
                           DMA_COUNT*sizeof(float),
                           1, 0, 1, 0, 1, 0);

      //
      // read the data
      //

      wr_idx = queue_add_write_index(queues[0], 1);
      packet_id = wr_idx % queues[0]->size;
      dispatch_packet_t *pkt2 = (dispatch_packet_t*)(queues[0]->base_address_vaddr) + packet_id;
      air_packet_nd_memcpy(pkt2, 0, 2+i, 0, 0+j, 4, 2, 
                           BRAM_ADDR+(4*DMA_COUNT*sizeof(float))+((2*i+j)*DMA_COUNT*sizeof(float)), 
                           DMA_COUNT*sizeof(float),
                           1, 0, 1, 0, 1, 0);
      signal_create(1, 0, NULL, (signal_t *)&pkt2->completion_signal);

      uint64_t s = queue_paddr_from_index(
          queues[0], (packet_id) * sizeof(dispatch_packet_t) + signal_offset);
      signals.push_back(s);

      p = pkt2;
    }
  }

  if (use_barrier) {
    //
    // Put a barrier AND packet in agent 1's queue
    //

    wr_idx2 = queue_add_write_index(queues[1], 1);
    packet_id2 = wr_idx2 % queues[1]->size;
    barrier_and_packet_t *barrier_pkt =
        (barrier_and_packet_t *)(queues[1]->base_address_vaddr) + packet_id2;
    air_packet_barrier_and(barrier_pkt, signals[0], signals[1], signals[2],
                           signals[3], 0);
    signal_create(1, 0, NULL, (signal_t *)&barrier_pkt->completion_signal);
  }

  //
  // send the data
  //

  wr_idx2 = queue_add_write_index(queues[1], 1);
  packet_id2 = wr_idx2 % queues[1]->size;
  dispatch_packet_t *pkt12 = (dispatch_packet_t*)(queues[1]->base_address_vaddr) + packet_id2;
  air_packet_nd_memcpy(pkt12, 0, 6, 1, 0, 4, 2, 
                       BRAM_ADDR+4*DMA_COUNT*sizeof(float), 
                       4*DMA_COUNT*sizeof(float), 
                       1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx2 = queue_add_write_index(queues[1], 1);
  packet_id2 = wr_idx2 % queues[1]->size;
  dispatch_packet_t *pkt22 = (dispatch_packet_t*)(queues[1]->base_address_vaddr) + packet_id2;
  air_packet_nd_memcpy(pkt22, 0, 6, 0, 0, 4, 2,
                       BRAM_ADDR+(8*DMA_COUNT*sizeof(float)),
                       4*DMA_COUNT*sizeof(float),
                       1, 0, 1, 0, 1, 0);

  // start herd 1 (waiting on barrier)
  air_queue_dispatch(queues[1], wr_idx2, pkt22);

  // start herd 0 
  air_queue_dispatch(queues[0], wr_idx, p);

  // wait for both to finish
  air_queue_wait(queues[0], p);
  air_queue_wait(queues[1], pkt22);

  int errors = 0;

  for (int i=0; i<4*DMA_COUNT; i++) {
    uint32_t d = bram_ptr[4*DMA_COUNT+i];
    if (d != (i+2)) {
      errors++;
      std::cout << "step 1 mismatch " << d << " != 2 + " << i << std::endl;
    }
  }

  for (int i=0; i<4*DMA_COUNT; i++) {
    uint32_t d = bram_ptr[8*DMA_COUNT+i];
    if (d != (i+3)) {
      errors++;
      std::cout << "step 2 mismatch " << d << " != 3 + " << i << std::endl;
    }
  }

  if (!errors) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  else {
    std::cout << "fail "
              << errors << "/" << 8*DMA_COUNT << std::endl;
    return -1;
  }

}
