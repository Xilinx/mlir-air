//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include "air_host.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define SHMEM_BASE 0x020100000000LL

#include "test_library.h"

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie.cpp"

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
  auto herd_col = 7;
  auto herd_row = 2;

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

  printCoreStatus(herd_col, herd_row, false, 0, 0);

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
  while (signal_wait_acquire((signal_t *)&herd_pkt->completion_signal,
                             HSA_SIGNAL_CONDITION_EQ, 0, 0x80000,
                             HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout on herd initialization!\n");
    printf("%x\n", herd_pkt->header);
    printf("%x\n", herd_pkt->type);
    printf("%x\n", (unsigned)herd_pkt->completion_signal);
  }

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();

  printCoreStatus(herd_col, herd_row, false, 0, 0);

  // We first write an ascending pattern into the area the AIE will write into
  for (uint32_t i=0; i<1024; i++) {
    mlir_write_buffer_beef_0_0(i, i+0x34431001);
  }
  printCoreStatus(herd_col, herd_row, false, 0, 0);

  printf("memory before:\n");
  for (uint32_t i=0; i<4; i++)
    printf("%d %x\n", i, mlir_read_buffer_beef_0_0(i));

  mlir_start_cores();

  int errors = 0;

  printf("memory after:\n");
  for (uint32_t i=0; i<4; i++)
    printf("%d %x\n", i, mlir_read_buffer_beef_0_0(i));

  u32 rb[4];
  for (uint32_t i=0; i<4; i++)
    rb[i] = mlir_read_buffer_beef_0_0(i);

  if (rb[0] != 0xdeadbeef)
    printf("Error %d: %08x != 0xdeadbeef\n",errors++, rb);
  
  if (rb[1] != 0xcafecafe)
    printf("Error %d: %08x != 0xcafecafe\n",errors++, rb);
  
  if (rb[2] != 0x000decaf)
    printf("Error %d: %08x != 0x000decaf\n",errors++, rb);
  
  if (rb[3] != 0x5a1ad000)
    printf("Error %d: %08x != 0x5a1ad000\n",errors++, rb);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", 4-errors, 4);
    return -1;
  }

}
