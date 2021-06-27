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
#include "test_library.h"

#include "air_host.h"
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

void printCoreStatus(int col, int row, bool PM, int mem, int trace) {

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
    printf("FIRST %d WORDS\n",mem);
    for (int i = 0; i < mem; i++) {
      u32 RegVal = XAieTile_DmReadWord(&(TileInst[col][row]), 0x1000 + i * 4);
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

  printCoreStatus(col, 2, true, SCRATCH_AREA, 0);
  
  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  // herd_setup packet
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(herd_pkt);
  herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  // Set up the worlds smallest herd at 7,2
  herd_pkt->arg[0]  = AIR_PKT_TYPE_HERD_INITIALIZE;
  herd_pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  herd_pkt->arg[0] |= (1L << 40);
  herd_pkt->arg[0] |= (7L << 32);
  herd_pkt->arg[0] |= (1L << 24);
  herd_pkt->arg[0] |= (2L << 16);
  
  herd_pkt->arg[1] = 0;  // Herd ID 0
  herd_pkt->arg[2] = 0;
  herd_pkt->arg[3] = 0;

  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&herd_pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  signal_store_release((signal_t*)&q->doorbell, wr_idx);

  // wait for packet completion
  while (signal_wait_aquire((signal_t*)&herd_pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout on herd initialization!\n");
    printf("%x\n", herd_pkt->header);
    printf("%x\n", herd_pkt->type);
    printf("%x\n", (unsigned)herd_pkt->completion_signal);
  }

  mlir_configure_cores();
  printCoreStatus(col, 2, false, SCRATCH_AREA, 0);
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();
  
  printCoreStatus(col, 2, false, SCRATCH_AREA, 0);
  // We first write an ascending pattern into the area the AIE will write into
  for (int i=0; i<SCRATCH_AREA; i++) {
    uint32_t d = i+1;
    mlir_write_buffer_buffer(i, d);
  }
  printCoreStatus(col, 2, false, SCRATCH_AREA, 0);

  // We wrote data, so lets tell the MicroBlaze to toggle the job lock 0
  // reserve another packet in the queue
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  // lock packet
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(lock_pkt);
  lock_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  // Release lock 0 in 0,0 with value 0
  lock_pkt->arg[0]  = AIR_PKT_TYPE_XAIE_LOCK;
  lock_pkt->arg[0] |= (AIR_ADDRESS_HERD_RELATIVE << 48);
  lock_pkt->arg[1]  = 0;
  lock_pkt->arg[2]  = 1;
  lock_pkt->arg[3]  = 1;

  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&lock_pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  signal_store_release((signal_t*)&q->doorbell, wr_idx);

  // wait for packet completion
  while (signal_wait_aquire((signal_t*)&lock_pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout on lock release!\n");
    printf("%x\n", lock_pkt->header);
    printf("%x\n", lock_pkt->type);
    printf("%x\n", (unsigned)lock_pkt->completion_signal);
  }

  printCoreStatus(col, 2, false, SCRATCH_AREA, 0);

  auto count = 0;
  while (!XAieTile_LockAcquire(&(TileInst[col][2]), 0, 0, 1000)) {
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  int errors = 0;
  ACDC_check("Check Result 0:", mlir_read_buffer_buffer(0), 0xdeadbeef);
  ACDC_check("Check Result 1:", mlir_read_buffer_buffer(1), 0xcafecafe);
  ACDC_check("Check Result 2:", mlir_read_buffer_buffer(2), 0x000decaf);
  ACDC_check("Check Result 3:", mlir_read_buffer_buffer(3), 0x5a1ad000);

  for (int i=4; i<SCRATCH_AREA; i++)
    ACDC_check("Check Result:", mlir_read_buffer_buffer(i), i+1);

  if (!errors) {
    printf("PASS!\n");
    return 0;
      }
  else {
    printf("fail %d/%d.\n", (SCRATCH_AREA-errors), SCRATCH_AREA);
    return -1;
      }
}
