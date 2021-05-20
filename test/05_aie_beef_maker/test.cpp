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

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define SCRATCH_AREA 8

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie.inc"

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
    printf("FIRST %d WORDS\n", mem);
    for (int i = 0; i < 8; i++) {
      u32 RegVal = XAieTile_DmReadWord(&(TileInst[col][row]), 0x1000 + (i * 4));
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

  printCoreStatus(col, 2, true, SCRATCH_AREA, 0);
  
  // cores - most of these calls do nothing - there's no routing or DMAs ...

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();
  
  printCoreStatus(col, 2, false, SCRATCH_AREA, 0);

  // We first write an ascending pattern into the area the AIE will write into
  for (int i=0; i<SCRATCH_AREA; i++) {
    uint32_t d = i+1;
    XAieTile_DmWriteWord(&(TileInst[col][2]), 0x1000 + (i*4), d);
  }
  printCoreStatus(col, 2, false, SCRATCH_AREA, 0);
  // We wrote data, so lets toggle the job lock 0
  XAieTile_LockRelease(&(TileInst[col][2]), 0, 0x1, 0);
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

  // If you are squeemish, look away now
  u32 rb;

  rb = XAieTile_DmReadWord(&(TileInst[col][2]), 0x1000 + (0x0 * sizeof(u32)));
  if (rb != 0xdeadbeef)
    printf("Error %d: %08x != 0xdeadbeef",errors++, rb);
  rb = XAieTile_DmReadWord(&(TileInst[col][2]), 0x1000 + (0x1 * sizeof(u32)));
  if (rb != 0xcafecafe)
    printf("Error %d: %08x != 0xcafecafe",errors++, rb);
  rb = XAieTile_DmReadWord(&(TileInst[col][2]), 0x1000 + (0x2 * sizeof(u32)));
  if (rb != 0x000decaf)
    printf("Error %d: %08x != 0x000decaf",errors++, rb);
  rb = XAieTile_DmReadWord(&(TileInst[col][2]), 0x1000 + (0x3 * sizeof(u32)));
  if (rb != 0x5a1ad000)
    printf("Error %d: %08x != 0x5a1ad000",errors++, rb);

  for (int i=4; i<SCRATCH_AREA; i++) {
    rb = XAieTile_DmReadWord(&(TileInst[col][2]),0x1000 + (i * sizeof(u32)));
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
