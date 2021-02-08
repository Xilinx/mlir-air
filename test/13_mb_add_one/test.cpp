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

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie.cpp"

}

int
main(int argc, char *argv[])
{
  uint64_t row = 0;
  uint64_t col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  uint32_t *bram_ptr;

  #define BRAM_ADDR 0x4000+0x020100000000LL
  #define DMA_COUNT 16

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = 0;
      bram_ptr[DMA_COUNT+i] = 0xdeface;
      //printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }
  }

  for (int i=0; i<DMA_COUNT*2; i++) {
    XAieTile_DmWriteWord(&(TileInst[col][2]), 0x1000+i*4, 0xdecaf);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(herd_pkt);
  herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  //
  // Set up a 1x3 herd starting 7,0
  //

  herd_pkt->arg[0]  = AIR_PKT_TYPE_HERD_INITIALIZE;
  herd_pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  herd_pkt->arg[0] |= (1L << 40);
  herd_pkt->arg[0] |= (7L << 32);
  herd_pkt->arg[0] |= (3L << 24);
  herd_pkt->arg[0] |= (0L << 16);
  
  herd_pkt->arg[1] = 0;  // Herd ID 0
  herd_pkt->arg[2] = 0;
  herd_pkt->arg[3] = 0;

  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&herd_pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  //signal_store_release((signal_t*)&q->doorbell, wr_idx);

  //
  // send the data
  //

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

  signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  //signal_store_release((signal_t*)&q->doorbell, wr_idx);

  //
  // read the data
  //

   wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  pkt->arg[0] = AIR_PKT_TYPE_SHIM_DMA_MEMCPY;
  pkt->arg[0] |= (row << 16);
  pkt->arg[0] |= (col << 32);
  pkt->arg[0] |= (flags << 48);
  
  direction = 0;
  channel = XAIEDMA_SHIM_CHNUM_S2MM0;

  pkt->arg[1] = burst_len;
  pkt->arg[1] |= (direction << 32);
  pkt->arg[1] |= (channel << 48);
  pkt->arg[2] = BRAM_ADDR+DMA_COUNT*sizeof(float);
  pkt->arg[3] = DMA_COUNT*sizeof(float);

  signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  signal_store_release((signal_t*)&q->doorbell, wr_idx);

  while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout!\n");
    printf("%x\n", pkt->header);
    printf("%x\n", pkt->type);
    printf("%x\n", pkt->completion_signal);
    break;
  }

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[DMA_COUNT+i];
    if (d != (i+2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT-errors), DMA_COUNT);
  }

}