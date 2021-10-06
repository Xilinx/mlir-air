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
#include "test_library.h"

#include "acdc_queue.h"
#include "hsa_defs.h"

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie_inc.cpp"
#undef TileInst
#undef TileDMAInst

}

int
main(int argc, char *argv[])
{
  uint64_t col = 7;
  uint64_t row = 0;

  xaie = air_init_libxaie1();

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  uint32_t *bram_ptr;

  #define BRAM_ADDR 0x020100000000LL
  #define DMA_COUNT 32

  // We're going to stamp over the memories
  for (int i=0; i<DMA_COUNT; i++) {
    mlir_write_buffer_buf72_0(i, 0xdeadbeef);
    mlir_write_buffer_buf74_0(i, 0xfeedf00d);
  }
  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");


  // Let's make a buffer that we can transfer in the same BRAM, after the queue of HSA packets
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd,  AIR_VCK190_SHMEM_BASE+0x4000);//BRAM_ADDR+(MB_QUEUE_SIZE*64));
  
  for (int i=0;i<DMA_COUNT;i++) {
    bram_ptr[i] = i;
    bram_ptr[DMA_COUNT+i] = i*2;
    bram_ptr[2*DMA_COUNT+i] = 0xf001ba11; // 50 years of hurt
    bram_ptr[3*DMA_COUNT+i] = 0x00051ade; // Cum on hear the noize
  }

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 5);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // This starts the copying to the tiles

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_a = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, 18, 1, 0, 4, 2, AIR_VCK190_SHMEM_BASE+0x4000, DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt_a);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_b = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_b, 0, 11, 1, 0, 4, 2, AIR_VCK190_SHMEM_BASE+0x4000+(DMA_COUNT*sizeof(float)), DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt_b);

  // This completes the copying to the tiles, let's move the pattern back

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_c = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_c, 0, 18, 0, 0, 4, 2, AIR_VCK190_SHMEM_BASE+0x4000+(2*DMA_COUNT*sizeof(float)), DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt_c);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_d = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_d, 0, 11, 0, 0, 4, 2, AIR_VCK190_SHMEM_BASE+0x4000+(3*DMA_COUNT*sizeof(float)), DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt_d);

  ACDC_print_dma_status(xaie->TileInst[7][2]);
  ACDC_print_dma_status(xaie->TileInst[7][4]);

  uint32_t errs = 0;
  // Let go check the tile memory
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_buf72_0(i);
    if (d != i) {
      printf("ERROR: buf72_0 id %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_buf74_0(i);
    if (d != i*2) {
      printf("ERROR: buf74_0 id %d Expected %08X, got %08X\n", i, i*2, d);
      errs++;
    }
  }
  // And the BRAM we updated
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[2*DMA_COUNT+i];;
    if (d != i) {
      printf("ERROR: buf72_0 copy id %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[3*DMA_COUNT+i];;
    if (d != i*2) {
      printf("ERROR: buf74_0 copy id %d Expected %08X, got %08X\n", i, i*2, d);
      errs++;
    }
  }

  if (errs == 0) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }

}
