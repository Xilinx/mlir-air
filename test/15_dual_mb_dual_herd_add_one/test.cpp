// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xaiengine.h>

#include "air_host.h"
#include "acdc_queue.h"
#include "hsa_defs.h"
#include "aie_inc.cpp"

int
main(int argc, char *argv[])
{
  uint64_t row  = 0;
  uint64_t col  = 7;
  uint64_t col2 = 34;

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
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = i+1;
      bram_ptr[DMA_COUNT+i]   = 0xdeface;
      bram_ptr[2*DMA_COUNT+i] = 0xdeface;
    }
  }

  for (int i=0; i<8; i++) {
    mlir_aie_write_buffer_ping_in(xaie, i, 0xabbaba10+i);
    mlir_aie_write_buffer_pong_in(xaie, i, 0xdeeded10+i);
    mlir_aie_write_buffer_ping_out(xaie, i, 0x12345610+i);
    mlir_aie_write_buffer_pong_out(xaie, i, 0x76543210+i);
    mlir_aie_write_buffer_ping_in2(xaie, i, 0xabbaba20+i);
    mlir_aie_write_buffer_pong_in2(xaie, i, 0xdeeded20+i);
    mlir_aie_write_buffer_ping_out2(xaie, i, 0x12345620+i);
    mlir_aie_write_buffer_pong_out2(xaie, i, 0x76543220+i);
  }

  // create the queues
  uint64_t* qaddrs = (uint64_t*)AIR_VCK190_SHMEM_BASE;
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, (uint64_t)&qaddrs[0]);
  assert(ret == 0 && "failed to create queue!");
  queue_t *q2 = nullptr;
  ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q2, (uint64_t)&qaddrs[1]);
  assert(ret == 0 && "failed to create queue!");

  //
  // Set up a 1x3 herd starting 7,0
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);
  //
  // Set up a 1x3 herd starting 34,0
  //
  uint64_t wr_idx2 = queue_add_write_index(q2, 1);
  uint64_t packet_id2 = wr_idx2 % q2->size;
  dispatch_packet_t *herd_pkt2 = (dispatch_packet_t*)(q2->base_address_vaddr) + packet_id2;
  air_packet_herd_init(herd_pkt2, 0, col2, 1, row, 3);
  air_queue_dispatch_and_wait(q2, wr_idx2, herd_pkt2);

  //
  // send the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt1 = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt1, 0, 7, 1, 0, 4, 2, BRAM_ADDR, DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt2 = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, 7, 0, 0, 4, 2, BRAM_ADDR+(DMA_COUNT*sizeof(float)), DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // send the data
  //

  wr_idx2 = queue_add_write_index(q2, 1);
  packet_id2 = wr_idx2 % q2->size;
  dispatch_packet_t *pkt12 = (dispatch_packet_t*)(q2->base_address_vaddr) + packet_id2;
  air_packet_nd_memcpy(pkt12, 0, 34, 1, 0, 4, 2, BRAM_ADDR, DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx2 = queue_add_write_index(q2, 1);
  packet_id2 = wr_idx2 % q2->size;
  dispatch_packet_t *pkt22 = (dispatch_packet_t*)(q2->base_address_vaddr) + packet_id2;
  air_packet_nd_memcpy(pkt22, 0, 34, 0, 0, 4, 2, BRAM_ADDR+(2*DMA_COUNT*sizeof(float)), DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);

  air_queue_dispatch(q, wr_idx, pkt2);
  air_queue_dispatch_and_wait(q2, wr_idx2, pkt22);
  air_queue_wait(q, pkt2);

  int errors = 0;

  for (int i=0; i<8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out(xaie, i);
    if (d0+1 != d2) {
      printf("1 mismatch ping %x != %x\n", d0, d2);
      errors++;
    }
    if (d1+1 != d3) {
      printf("1 mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[DMA_COUNT+i];
    if (d != (i+2)) {
      errors++;
      printf("1 mismatch %x != 2 + %x\n", d, i);
    }
  }
  for (int i=0; i<8; i++) {
    uint32_t d0 = mlir_aie_read_buffer_ping_in2(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_pong_in2(xaie, i);
    uint32_t d2 = mlir_aie_read_buffer_ping_out2(xaie, i);
    uint32_t d3 = mlir_aie_read_buffer_pong_out2(xaie, i);
    if (d0+1 != d2) {
      printf("2 mismatch ping %x != %x\n", d0, d2);
      errors++;
    }
    if (d1+1 != d3) {
      printf("2 mismatch pong %x != %x\n", d1, d3);
      errors++;
    }
  }

  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[2*DMA_COUNT+i];
    if (d != (i+2)) {
      errors++;
      printf("2 mismatch %x != 2 + %x\n", d, i);
    }
  }
  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, 2*(DMA_COUNT+4*8));
    return -1;
  }

}
