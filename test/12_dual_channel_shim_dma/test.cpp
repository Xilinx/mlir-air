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

#include "aie_inc.cpp"

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
  auto col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);


  printCoreStatus(col, 2);
  printCoreStatus(col, 4);
  
  // cores
  //
  mlir_configure_cores();

  // configure switchboxes
  //
  mlir_configure_switchboxes();
  //XAieTile_ShimStrmMuxConfig(&(TileInst[col][0]), XAIETILE_SHIM_STRM_MUX_SOUTH7, XAIETILE_SHIM_STRM_MUX_DMA);

  // locks
  //
  mlir_initialize_locks();

  // dmas
  //

  mlir_configure_dmas();

  XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

  #define BRAM_ADDR (0x4000+0x020100000000LL)
  #define DMA_COUNT 16

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    // + 1 ascending pattern
    for (int i=0; i<DMA_COUNT*2; i++) {
      bram_ptr[i] = i+1;
      //printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }
    // + 2 ascending pattern
    for (int i=DMA_COUNT*2; i<DMA_COUNT*4; i++) {
      bram_ptr[i] = (i-DMA_COUNT*2)+2;
      //printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }

    // Stomp
    for (int i=DMA_COUNT*4; i<DMA_COUNT*8; i++) {
      bram_ptr[i] = 0x5a1ad;
      //printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }


  }

  // We're going to stamp over the local memories where the data is going to end up
  for (int i=0; i<DMA_COUNT*2; i++) {
    XAieTile_DmWriteWord(&(TileInst[col][2]), 0x1000 + i*4, 0xdefaced0);
  }
  // We're going to stamp over the memories
  for (int i=0; i<DMA_COUNT*2; i++) {
    XAieTile_DmWriteWord(&(TileInst[col][4]), 0x1000 + i*4, 0x0defaced);
  }

  auto burstlen = 4;
  XAieDma_ShimInitialize(&(TileInst[col][0]), &ShimDmaInst1);
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)BRAM_ADDR), LOW_ADDR((u64)BRAM_ADDR), sizeof(u32) * DMA_COUNT*2);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, 1);

  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 2, HIGH_ADDR((u64)BRAM_ADDR+(2*DMA_COUNT*sizeof(u32))), LOW_ADDR((u64)BRAM_ADDR+(2*DMA_COUNT*sizeof(u32))), sizeof(u32) * DMA_COUNT * 2);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 2 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 2);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S1, 2);

  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 3, HIGH_ADDR((u64)BRAM_ADDR+(4*DMA_COUNT*sizeof(u32))), LOW_ADDR((u64)BRAM_ADDR+(4*DMA_COUNT*sizeof(u32))), sizeof(u32) * DMA_COUNT*2);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 3 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 3);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, 3);

  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 4, HIGH_ADDR((u64)BRAM_ADDR+(6*DMA_COUNT*sizeof(u32))), LOW_ADDR((u64)BRAM_ADDR+(6*DMA_COUNT*sizeof(u32))), sizeof(u32) * DMA_COUNT*2);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 4 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 4);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM1, 4);

  auto ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S1);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_S2MM0);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_S2MM1);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S1, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM1, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("MM2S0 %d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }
  count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S1)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("MM2S1 %d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_S2MM0)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("S2MM0 %d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_S2MM1)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("S2MM1 %d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  printCoreStatus(col, 2);
  printCoreStatus(col, 4);
  
  int errors = 0;
  for (int i=0; i<DMA_COUNT*2; i++) {
    uint32_t d = XAieTile_DmReadWord(&(TileInst[col][2]), 0x1000+i*4);
    if (d != (i+1)) {
      errors++;
      printf("Tile Memory[%d][%d] Address %04X mismatch %x != 1 + %x\n", col, 2, 0x1000+i*4,d, i);
    }
  }
  
  for (int i=0; i<DMA_COUNT*2; i++) {
    uint32_t d = XAieTile_DmReadWord(&(TileInst[col][4]), 0x1000+i*4);
    if (d != (i+2)) {
      errors++;
      printf("Tile Memory[%d][%d] Address %04X mismatch %x != 2 + %x\n", col, 4, 0x1000+i*4,d, i);
    }
  }

  for (int i=0; i<DMA_COUNT*2; i++) {
    uint32_t d = bram_ptr[DMA_COUNT*4+i];
    if (d != (i+1)) {
      errors++;
      printf("Ext Memory offset %04X mismatch %x != 1 + %x\n", DMA_COUNT*4+i,d, i);
    }
  }

  for (int i=0; i<DMA_COUNT*2; i++) {
    uint32_t d = bram_ptr[DMA_COUNT*6+i];
    if (d != (i+2)) {
      errors++;
      printf("Ext Memory offset %04X mismatch %x != 2 + %x\n", DMA_COUNT*6+i,d, i);
    }
  }


  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT*4-errors), DMA_COUNT*4);
  }

}
