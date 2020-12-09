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

#include "acdc_queue.h"
#include "hsa_defs.h"


#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define SCRATCH_AREA 8

#define SHMEM_BASE 0x020100000000LL

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie_inc.cpp"

}

hsa_status_t queue_create(uint32_t size, uint32_t type, queue_t **queue)
{
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  uint64_t *bram_ptr = (uint64_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, SHMEM_BASE);
  // I have no idea if this does anything
  __clear_cache((void*)bram_ptr, (void*)(bram_ptr+0x1000));
  //for (int i=0; i<20; i++)
  //  printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);

  printf("Opened shared memory paddr: %p vaddr: %p\n", SHMEM_BASE, bram_ptr);
  uint64_t q_paddr = bram_ptr[0];
  uint64_t q_offset = q_paddr - SHMEM_BASE;
  queue_t *q = (queue_t*)( ((size_t)bram_ptr) + q_offset );
  printf("Queue location at paddr: %p vaddr: %p\n", bram_ptr[0], q);

  if (q->id !=  0xacdc) {
    printf("%s error invalid id %x\n", __func__, q->id);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->size != size) {
    printf("%s error size mismatch %d\n", __func__, q->size);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->type != type) {
    printf("%s error type mismatch %d\n", __func__, q->type);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  uint64_t base_address_offset = q->base_address - SHMEM_BASE;
  q->base_address_vaddr = ((size_t)bram_ptr) + base_address_offset;

  *queue = q;
  return HSA_STATUS_SUCCESS;
}


void printCoreStatus(int col, int row, bool PM, bool mem, int trace) {

  u32 status, coreTimerLow, PC, LR, SP, locks, R0, R4;
  status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x032004);
  coreTimerLow = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0340F8);
  PC = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x00030280);
  LR = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x000302B0);
  SP = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x000302A0);
  locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
  u32 trace_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x000140D8);
  R0 = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x00030000);
  R4 = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x00030040);
  printf("Core [%d, %d] status is %08X, timer is %u, PC is %d, locks are %08X, LR is %08X, SP is %08X, R0 is %08X,R4 is %08X\n",col, row, status, coreTimerLow, PC, locks, LR, SP, R0, R4);
  printf("Core [%d, %d] trace status is %08X\n",col, row, trace_status);
  for (int lock=0;lock<16;lock++) {
    u32 two_bits = (locks >> (lock*2)) & 0x3;
    if (two_bits) {
      printf("Lock %d: ", lock);
      u32 acquired = two_bits & 0x1;
      u32 value = two_bits & 0x2;
      if (acquired)
	printf("Acquired ");
      printf(value?"1":"0");
      printf("\n");
    }
  }
  // Read the warning above!!!!!
  if (PM)
    for (int i=0;i<40;i++)
      printf("PM[%d]: %08X\n",i*4, XAieGbl_Read32(TileInst[col][row].TileAddr + 0x00020000 + i*4));
  if (mem) {
    printf("FIRST 8 WORDS\n");
    for (int i = 0; i < 8; i++) {
      u32 RegVal = XAieTile_DmReadWord(&(TileInst[col][row]), i * 4);
      printf("memory value %d : %08X %f\n", i, RegVal, *(float *)(&RegVal));
    }
  }
}


int
main(int argc, char *argv[])
{
  auto col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  // reset cores and locks
  for (int i = 1; i <= XAIE_NUM_ROWS; i++) {
    for (int j = 0; j < XAIE_NUM_COLS; j++) {
      XAieTile_CoreControl(&(TileInst[j][i]), XAIE_DISABLE, XAIE_ENABLE);
      for (int l=0; l<16; l++)
        XAieTile_LockRelease(&(TileInst[j][i]), l, 0x0, 0);
    }
  }

  printCoreStatus(col, 2, true, true, 0);
  
  // create the queue
  queue_t *q = nullptr;
  auto ret = queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q);
  assert(ret == 0 && "failed to create queue!");



  // cores
  //
  //  mlir_initialize_cores();

  mlir_configure_cores();
  //XAieTile_CoreControl(&(TileInst[col][2]), XAIE_DISABLE, XAIE_DISABLE);
  printCoreStatus(col, 2, false, true, 0);
  
  // configure switchboxes
  //
  mlir_configure_switchboxes();
  
  // locks
  //
  mlir_initialize_locks();

  // dmas
  //

  mlir_configure_dmas();

  mlir_start_cores();
  
  printCoreStatus(col, 2, false, true, 0);

  // We first write an ascending pattern into the area the AIE will write into
  for (int i=0; i<SCRATCH_AREA; i++) {
    uint32_t d = i+1;
    XAieTile_DmWriteWord(&(TileInst[col][2]), i*4, d);
  }
  printCoreStatus(col, 2, false, true, 0);

  // We wrote data, so lets tell the MicroBlaze to toggle the job lock 0
  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  // setup packet
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  // release lock 0 with value 1
  pkt->arg[0] = AIR_PKT_TYPE_XAIE_LOCK;
  pkt->arg[1] = 0;  // Which lock?
  pkt->arg[2] = 1;  // Acquire = 0, Release = 1
  pkt->arg[3] = 1;  // What value to use
  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  signal_store_release((signal_t*)&q->doorbell, wr_idx);

  // wait for packet completion
  while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout!\n");
    printf("%x\n", pkt->header);
    printf("%x\n", pkt->type);
    printf("%x\n", pkt->completion_signal);
  }

  printCoreStatus(col, 2, false, true, 0);

  auto count = 0;
  while (!XAieTile_LockAcquire(&(TileInst[col][2]), 0, 0, 1000)) {
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  int errors = 0;

  // If you are squeemish, look away now
  u32 rb;

  rb = XAieTile_DmReadWord(&(TileInst[col][2]), 0x0 * sizeof(u32));
  if (rb != 0xdeadbeef)
    printf("Error %d: %08x != 0xdeadbeef",errors++, rb);
  rb = XAieTile_DmReadWord(&(TileInst[col][2]), 0x1 * sizeof(u32));
  if (rb != 0xcafecafe)
    printf("Error %d: %08x != 0xcafecafe",errors++, rb);
  rb = XAieTile_DmReadWord(&(TileInst[col][2]), 0x2 * sizeof(u32));
  if (rb != 0x000decaf)
    printf("Error %d: %08x != 0x000decaf",errors++, rb);
  rb = XAieTile_DmReadWord(&(TileInst[col][2]), 0x3 * sizeof(u32));
  if (rb != 0x5a1ad000)
    printf("Error %d: %08x != 0x5a1ad000",errors++, rb);

  for (int i=4; i<SCRATCH_AREA; i++) {
    rb = XAieTile_DmReadWord(&(TileInst[col][2]), i * sizeof(u32));
    if (rb != i+1)
      printf("Error %d: %08x != %08x\n", errors++, rb, i+1);
  }

  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (SCRATCH_AREA-errors), SCRATCH_AREA);
  }

}
