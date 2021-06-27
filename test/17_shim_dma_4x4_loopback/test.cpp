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

#define NUM_SHIM_DMAS 8

int main(int argc, char *argv[]) {
  unsigned cols[NUM_SHIM_DMAS] = {2,3,6,7,10,11,18,19};

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();

  XAieDma_Shim ShimDmaInst[NUM_SHIM_DMAS];

  uint32_t *input_bram_ptr;
  uint32_t *output_bram_ptr;

  #define INPUT_BRAM_ADDR (0x4000+0x020100000000LL)
  #define OUTPUT_BRAM_ADDR (0xC000+0x020100000000LL)
  #define DMA_COUNT 16

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    input_bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, INPUT_BRAM_ADDR);
    output_bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, OUTPUT_BRAM_ADDR);
    // + 1 ascending pattern
    for (int i=0; i<DMA_COUNT*4*NUM_SHIM_DMAS; i++) {
      input_bram_ptr[i] = i+1;
    }

    // Stomp
    for (int i=0; i<DMA_COUNT*4*NUM_SHIM_DMAS; i++) {
      output_bram_ptr[i] = 0x5a1ad;
    }
  }

  // We're going to stamp over the local memories where the data is going to end up
  for (int col=8; col<=11; col++)
    for (int row=3; row<=6; row++)
      ACDC_clear_tile_memory(TileInst[col][row]);

  auto burstlen = 4;
  for (int col=0;col<NUM_SHIM_DMAS;col++) {
    printf("Initializing %d shim dma in col %d\n",col, cols[col]);
    XAieDma_ShimInitialize(&(TileInst[cols[col]][0]), &(ShimDmaInst[col]));
    XAieDma_ShimBdSetAddr(&(ShimDmaInst[col]), 1, HIGH_ADDR((u64)INPUT_BRAM_ADDR+(col*DMA_COUNT*4*sizeof(u32))), LOW_ADDR((u64)INPUT_BRAM_ADDR+(col*DMA_COUNT*4*sizeof(u32))), sizeof(u32) * DMA_COUNT*2);
    XAieDma_ShimBdSetAxi(&(ShimDmaInst[col]), 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
    XAieDma_ShimBdWrite(&(ShimDmaInst[col]), 1);
    XAieDma_ShimSetStartBd((&(ShimDmaInst[col])), XAIEDMA_SHIM_CHNUM_MM2S0, 1);

    XAieDma_ShimBdSetAddr(&(ShimDmaInst[col]), 2, HIGH_ADDR((u64)INPUT_BRAM_ADDR+(col*DMA_COUNT*4*sizeof(u32))+(2*DMA_COUNT*sizeof(u32))), LOW_ADDR((u64)INPUT_BRAM_ADDR+(col*DMA_COUNT*4*sizeof(u32))+(2*DMA_COUNT*sizeof(u32))), sizeof(u32) * DMA_COUNT * 2);
    XAieDma_ShimBdSetAxi(&(ShimDmaInst[col]), 2 , 0, burstlen, 0, 0, XAIE_ENABLE);
    XAieDma_ShimBdWrite(&(ShimDmaInst[col]), 2);
    XAieDma_ShimSetStartBd((&(ShimDmaInst[col])), XAIEDMA_SHIM_CHNUM_MM2S1, 2);

    XAieDma_ShimBdSetAddr(&(ShimDmaInst[col]), 3, HIGH_ADDR((u64)OUTPUT_BRAM_ADDR+(col*DMA_COUNT*4*sizeof(u32))), LOW_ADDR((u64)OUTPUT_BRAM_ADDR+(col*DMA_COUNT*4*sizeof(u32))), sizeof(u32) * DMA_COUNT*2);
    XAieDma_ShimBdSetAxi(&(ShimDmaInst[col]), 3 , 0, burstlen, 0, 0, XAIE_ENABLE);
    XAieDma_ShimBdWrite(&(ShimDmaInst[col]), 3);
    XAieDma_ShimSetStartBd((&(ShimDmaInst[col])), XAIEDMA_SHIM_CHNUM_S2MM0, 3);

    XAieDma_ShimBdSetAddr(&(ShimDmaInst[col]), 4, HIGH_ADDR((u64)OUTPUT_BRAM_ADDR+(col*DMA_COUNT*4*sizeof(u32))+(2*DMA_COUNT*sizeof(u32))), LOW_ADDR((u64)OUTPUT_BRAM_ADDR+(col*DMA_COUNT*4*sizeof(u32))+(2*DMA_COUNT*sizeof(u32))), sizeof(u32) * DMA_COUNT * 2);
    XAieDma_ShimBdSetAxi(&(ShimDmaInst[col]), 4 , 0, burstlen, 0, 0, XAIE_ENABLE);
    XAieDma_ShimBdWrite(&(ShimDmaInst[col]), 4);
    XAieDma_ShimSetStartBd((&(ShimDmaInst[col])), XAIEDMA_SHIM_CHNUM_S2MM1, 4);

    auto ret = XAieDma_ShimPendingBdCount(&(ShimDmaInst[col]), XAIEDMA_SHIM_CHNUM_MM2S0);
    if (ret)
      printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

    ret = XAieDma_ShimPendingBdCount(&(ShimDmaInst[col]), XAIEDMA_SHIM_CHNUM_MM2S1);
    if (ret)
      printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

    ret = XAieDma_ShimPendingBdCount(&(ShimDmaInst[col]), XAIEDMA_SHIM_CHNUM_S2MM0);
    if (ret)
      printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

    ret = XAieDma_ShimPendingBdCount(&(ShimDmaInst[col]), XAIEDMA_SHIM_CHNUM_S2MM1);
    if (ret)
      printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

    XAieDma_ShimChControl((&(ShimDmaInst[col])), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
    XAieDma_ShimChControl((&(ShimDmaInst[col])), XAIEDMA_SHIM_CHNUM_MM2S1, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
    XAieDma_ShimChControl((&(ShimDmaInst[col])), XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
    XAieDma_ShimChControl((&(ShimDmaInst[col])), XAIEDMA_SHIM_CHNUM_S2MM1, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
  }
  /*
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
  */

  for (int col=0;col<NUM_SHIM_DMAS;col++) {
    printCoreStatus(cols[col], 1);
    printCoreStatus(cols[col], 2);
  }


  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_a0(i);
    ACDC_check("Check Result a0:", d, i+1);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_a1(i);
    ACDC_check("Check Result a1:", d, i+DMA_COUNT+1);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_f0(i);
    ACDC_check("Check Result f0:", d, i+DMA_COUNT*10+1);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_f1(i);
    ACDC_check("Check Result f1:", d, i+DMA_COUNT*11+1);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_p0(i);
    ACDC_check("Check Result p0:", d, i+DMA_COUNT*30+1);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_p1(i);
    ACDC_check("Check Result p1:", d, i+DMA_COUNT*31+1);
  }

  // Let's just compare the input and output buffers
  for (int i=0; i<DMA_COUNT*4*NUM_SHIM_DMAS; i++) {
    ACDC_check("Check Result:", input_bram_ptr[i], output_bram_ptr[i]);
  }

/*
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
*/
  

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT*8*NUM_SHIM_DMAS-errors), DMA_COUNT*8*NUM_SHIM_DMAS);
    return -1;
  }

}
