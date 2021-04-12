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

void printCoreStatus(int col, int row) {

  u32 status, coreTimerLow, locks;
  status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x032004);
  coreTimerLow = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0340F8);
  locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
  printf("Core [%d, %d] status is %08X, timer is %u, locks are %08X\n",col, row, status, coreTimerLow, locks);
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
}


int
main(int argc, char *argv[])
{
  auto shim_col = 2;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

  #define BRAM_ADDR (0x4000+0x020100000000LL)
  #define IMAGE_WIDTH 32
  #define IMAGE_HEIGHT 16
  #define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

  #define TILE_WIDTH 16
  #define TILE_HEIGHT 8
  #define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<IMAGE_SIZE; i++) {
      bram_ptr[i] = i;
      bram_ptr[i+IMAGE_SIZE] = 0x00defaced;
    }
  }

  for (int i=0; i<TILE_SIZE; i++) {
    uint32_t d = i+1;
    mlir_write_buffer_scratch_0_0(i,0xfadefade);
  }

  // Fire up the AIE array

  mlir_configure_cores();
  mlir_configure_dmas();
  mlir_initialize_locks();
  mlir_configure_switchboxes();

  mlir_start_cores();

  XAieDma_ShimInitialize(&(TileInst[shim_col][0]), &ShimDmaInst1);
  XAieDma_ShimBdClearAll((&ShimDmaInst1));
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  for (int row = 0; row < TILE_HEIGHT; row++) {
    auto burstlen = 4;
    u64 start_addr = (u64)BRAM_ADDR + row*IMAGE_WIDTH*sizeof(u32);
    //printf("row %d : addr %016lX\n", row, start_addr);
    u32 bd = row % 4;
    XAieDma_ShimBdSetAddr(&ShimDmaInst1, bd, HIGH_ADDR(start_addr), LOW_ADDR(start_addr), sizeof(u32) * TILE_WIDTH);
    XAieDma_ShimBdSetAxi(&ShimDmaInst1, bd, 0, burstlen, 0, 0, XAIE_ENABLE);
    XAieDma_ShimBdWrite(&ShimDmaInst1, bd);
    XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, bd); 
  }

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  // Go look at the scratch buffer
  int errors = 0;
  for (int i=0;i<TILE_SIZE;i++) {
    u32 rb = mlir_read_buffer_scratch_0_0(i);
    u32 row = i / TILE_WIDTH;
    u32 col = i % TILE_WIDTH;
    u32 orig_index = row * IMAGE_WIDTH + col;
    if (!(rb == orig_index)) {
      printf("SB %d [%d, %d] should be %08X, is %08X\n", i, col, row, orig_index, rb);
      errors++;
    }
  }

  // Let's get it back!

  for (int row = 0; row < TILE_HEIGHT; row++) {
    auto burstlen = 4;
    u64 start_addr = (u64)BRAM_ADDR + (IMAGE_SIZE + row*IMAGE_WIDTH)*sizeof(u32);
    //printf("row %d : addr %016lX\n", row, start_addr);
    u32 bd = row % 4;

    XAieDma_ShimBdSetAddr(&ShimDmaInst1, bd, HIGH_ADDR(start_addr), LOW_ADDR(start_addr), sizeof(u32) * TILE_WIDTH);
    XAieDma_ShimBdSetAxi(&ShimDmaInst1, bd, 0, burstlen, 0, 0, XAIE_ENABLE);
    XAieDma_ShimBdWrite(&ShimDmaInst1, bd);
    XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, bd); 
  }

  count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_S2MM0)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }


  // Now look at the image, should have the top left filled in
  for (int i=0;i<IMAGE_SIZE;i++) {
    u32 rb = bram_ptr[i+IMAGE_SIZE]; // = 0x00defaced;

    u32 row = i / IMAGE_WIDTH;
    u32 col = i % IMAGE_WIDTH;

    if ((row < TILE_HEIGHT) && (col < TILE_WIDTH)) {
      if (!(rb == i)) {
        printf("IM %d [%d, %d] should be %08X, is %08X\n", i, col, row, i, rb);
        errors++;
      }
    }
    else {
      if (rb != 0x00defaced) {
        printf("IM %d [%d, %d] should be 0xdefaced, is %08X\n", i, col, row, rb);
        errors++;
      }
    }
  }

  // Now clean up

  XAieDma_ShimBdClearAll((&ShimDmaInst1));
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE, XAIE_DISABLE, XAIE_DISABLE);
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_DISABLE);

  for (int bd=0;bd<16;bd++) {
    // Take no prisoners.  No regerts
    // Overwrites the DMA_BDX_Control registers
    u32 rb = XAieGbl_Read32(TileInst[shim_col][0].TileAddr + 0x0001D008+(bd*0x14));
    printf("Before : bd%x control is %08X\n", bd, rb);
    XAieGbl_Write32(TileInst[shim_col][0].TileAddr + 0x0001D008+(bd*0x14), 0x0);

  }

  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (TILE_SIZE+IMAGE_SIZE-errors), TILE_SIZE+IMAGE_SIZE);
  }

}
