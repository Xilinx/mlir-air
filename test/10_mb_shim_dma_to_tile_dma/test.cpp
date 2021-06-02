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

#define SHMEM_BASE 0x020100000000LL

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie.inc"
#undef TileInst
#undef TileDMAInst

}

int
main(int argc, char *argv[])
{
  uint64_t col = 7;
  uint64_t row = 0;

  xaie = air_init_libxaie1();

  ACDC_print_dma_status(xaie->TileInst[7][2]);


  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

  #define BRAM_ADDR 0x020100000000LL
  #define DMA_COUNT 512

  // We're going to stamp over the memory
  for (int i=0; i<DMA_COUNT; i++)
    mlir_write_buffer_b0(i, 0xdeadbeef);

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, 0x020100000000LL);
  assert(ret == 0 && "failed to create queue!");


  // Let's make a buffer that we can transfer in the same BRAM, after the queue of HSA packets
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR+(MB_QUEUE_SIZE*64));
  
  for (int i=0;i<DMA_COUNT;i++) {
    bram_ptr[i] = i;
  }

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  dispatch_packet_t *shim_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(shim_pkt);
  shim_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  shim_pkt->arg[0]  = AIR_PKT_TYPE_DEVICE_INITIALIZE;
  shim_pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  shim_pkt->arg[0] |= ((uint64_t)XAIE_NUM_COLS << 40);

  air_queue_dispatch_and_wait(q, wr_idx, shim_pkt);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  pkt->arg[0] = AIR_PKT_TYPE_ND_MEMCPY;

  pkt->arg[0] |= (col << 32);
  
  uint64_t burst_len = 4;
  uint64_t direction = 1;
  uint64_t channel = 0;

  pkt->arg[0] |= burst_len << 52;
  pkt->arg[0] |= (direction << 60);
  pkt->arg[0] |= (channel << 24);
  pkt->arg[0] |= (2 << 16);

  pkt->arg[1]  = BRAM_ADDR+(MB_QUEUE_SIZE*64);
  pkt->arg[2]  = DMA_COUNT*sizeof(float);
  pkt->arg[2] |= (1L<<32);
  pkt->arg[3]  = 1;
  pkt->arg[3] |= (1L<<32);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);


  ACDC_print_dma_status(xaie->TileInst[7][2]);
  uint32_t errs = 0;
  // Let go check the tile memory
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_b0(i);
    if (d != i) {
      printf("ERROR: id %d Expected %08X, got %08X\n", i, i, d);
      errs++;
    }
  }

  if (errs == 0)
    printf("PASS!\n");
  else
    printf("fail.\n");

}
