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

#include "air_host.h"

#include "aie_inc.cpp"

#define NUM_SHIM_DMAS 8

int main(int argc, char *argv[]) {
  unsigned cols[NUM_SHIM_DMAS] = {2,3,6,7,10,11,18,19};

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);

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
      mlir_aie_clear_tile_memory(xaie, col, row);

  auto burstlen = 4;
  for (int col=0;col<NUM_SHIM_DMAS;col++) {
    printf("Initializing %d shim dma in col %d\n",col, cols[col]);
    XAieDma_ShimInitialize(&(xaie->TileInst[cols[col]][0]), &(ShimDmaInst[col]));
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


  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_a0(xaie, i);
    mlir_aie_check("Check Result a0:", d, i+1,errors);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_a1(xaie, i);
    mlir_aie_check("Check Result a1:", d, i+DMA_COUNT+1,errors);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_f0(xaie, i);
    mlir_aie_check("Check Result f0:", d, i+DMA_COUNT*10+1,errors);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_f1(xaie, i);
    mlir_aie_check("Check Result f1:", d, i+DMA_COUNT*11+1,errors);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_p0(xaie, i);
    mlir_aie_check("Check Result p0:", d, i+DMA_COUNT*30+1,errors);
  }
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_p1(xaie, i);
    mlir_aie_check("Check Result p1:", d, i+DMA_COUNT*31+1,errors);
  }

  // Let's just compare the input and output buffers
  for (int i=0; i<DMA_COUNT*4*NUM_SHIM_DMAS; i++) {
    mlir_aie_check("Check Result:", input_bram_ptr[i], output_bram_ptr[i],errors);
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
