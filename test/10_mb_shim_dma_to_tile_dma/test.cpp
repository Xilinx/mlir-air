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
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

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
  pkt->arg[0] = AIR_PKT_TYPE_SHIM_DMA_MEMCPY;
  pkt->arg[0] |= (row << 16);
  pkt->arg[0] |= (col << 32);
  uint64_t flags = 0x1;
  pkt->arg[0] |= (flags << 48);
  
  uint32_t burst_len = 4;
  uint64_t direction = 1;
  uint64_t channel = XAIEDMA_SHIM_CHNUM_MM2S0;

  pkt->arg[1] = burst_len;
  pkt->arg[1] |= (direction << 32);
  pkt->arg[1] |= (channel << 48);
  pkt->arg[2] = BRAM_ADDR;
  pkt->arg[3] = DMA_COUNT*sizeof(float);

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  // we copied the start of the shared bram into tile memory,
  // fish out the queue id and check it
  for (int i=0; i<30; i++) {
    uint32_t d = mlir_read_buffer_b0(i);
    printf("ID %x\n", d);
  }

  uint32_t d = mlir_read_buffer_b0(24);
  if (d == 0xacdc)
    printf("PASS!\n");
  else
    printf("fail.\n");

}
