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

void printDMAStatus(int col, int row) {


  u32 dma_mm2s_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF10);
  u32 dma_s2mm_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF00);
  u32 dma_mm2s_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DE10);
  u32 dma_s2mm_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DE00);
  u32 dma_bd0_a       = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D000); 
  u32 dma_bd0_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D018);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/ctrl is %08X %08X, s2mm_status is %08X %08X, BD0_Addr_A is %08X, BD0_control is %08X\n",col, row, dma_mm2s_status, dma_mm2s_control, dma_s2mm_status, dma_s2mm_control, dma_bd0_a, dma_bd0_control);
  for (int bd=0;bd<8;bd++) {
      u32 dma_bd_addr_a        = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D000 + (0x20*bd));
      u32 dma_bd_control       = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D018 + (0x20*bd));
    if (dma_bd_control & 0x80000000) {
      printf("BD %d valid\n",bd);
      int current_s2mm_ch0 = (dma_s2mm_status >> 16) & 0xf;  
      int current_s2mm_ch1 = (dma_s2mm_status >> 20) & 0xf;  
      int current_mm2s_ch0 = (dma_mm2s_status >> 16) & 0xf;  
      int current_mm2s_ch1 = (dma_mm2s_status >> 20) & 0xf;  

      if (s2mm_ch0_running && bd == current_s2mm_ch0) {
        printf(" * Current BD for s2mm channel 0\n");
      }
      if (s2mm_ch1_running && bd == current_s2mm_ch1) {
        printf(" * Current BD for s2mm channel 1\n");
      }
      if (mm2s_ch0_running && bd == current_mm2s_ch0) {
        printf(" * Current BD for mm2s channel 0\n");
      }
      if (mm2s_ch1_running && bd == current_mm2s_ch1) {
        printf(" * Current BD for mm2s channel 1\n");
      }

      if (dma_bd_control & 0x08000000) {
        u32 dma_packet = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D010 + (0x20*bd));
        printf("   Packet mode: %02X\n",dma_packet & 0x1F);
      }
      int words_to_transfer = 1+(dma_bd_control & 0x1FFF);
      int base_address = dma_bd_addr_a  & 0x1FFF;
      printf("   Transfering %d 32 bit words to/from %05X\n",words_to_transfer, base_address);

      printf("   ");
      for (int w=0;w<4; w++) {
        printf("%08X ",XAieTile_DmReadWord(&(TileInst[col][row]), (base_address+w) * 4));
      }
      printf("\n");
      if (dma_bd_addr_a & 0x40000) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        printf("   Acquires lock %d ",lock_id);
        if (dma_bd_addr_a & 0x10000) 
          printf("with value %d ",(dma_bd_addr_a >> 17) & 0x1);

        printf("currently ");
        u32 locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
        u32 two_bits = (locks >> (lock_id*2)) & 0x3;
        if (two_bits) {
          u32 acquired = two_bits & 0x1;
          u32 value = two_bits & 0x2;
          if (acquired)
            printf("Acquired ");
          printf(value?"1":"0");
        }
        else printf("0");
        printf("\n");

      }
      if (dma_bd_control & 0x30000000) { // FIFO MODE
        int FIFO = (dma_bd_control >> 28) & 0x3;
          u32 dma_fifo_counter = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF20);				
        printf("   Using FIFO Cnt%d : %08X\n",FIFO, dma_fifo_counter);
      }
    }

  }

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

  // cores
  //
  mlir_configure_cores();

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

  XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

  #define BRAM_ADDR 0x4000+0x020100000000LL
  #define DMA_COUNT 16

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = i+1;
      bram_ptr[DMA_COUNT+i] = 0xdeface;
      //printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }
  }

  for (int i=0; i<DMA_COUNT*2; i++) {
    XAieTile_DmWriteWord(&(TileInst[col][2]), i*4, 0xdecaf);
  }


  printCoreStatus(7, 2, false, DMA_COUNT*2, 0);
  printDMAStatus(7, 2);

  auto burstlen = 4;
  XAieDma_ShimInitialize(&(TileInst[col][0]), &ShimDmaInst1);
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)BRAM_ADDR), LOW_ADDR((u64)BRAM_ADDR), sizeof(u32) * DMA_COUNT);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, 1);

  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 2, HIGH_ADDR((u64)BRAM_ADDR), LOW_ADDR((u64)BRAM_ADDR+DMA_COUNT*sizeof(u32)), sizeof(u32) * DMA_COUNT);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 2 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 2);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, 2);


  auto ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_S2MM0);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0) || XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0) ) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }


  // We want lock 2 and 3 to be released with the value 0, so an 0 in the second nibble
  count = 0;
  while ((XAieGbl_Read32(TileInst[col][2].TileAddr + 0x000302B0) & 0x0f0) != 0x000) {
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  printCoreStatus(7, 2, false, DMA_COUNT*2, 0);
  printDMAStatus(7, 2);

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    //uint32_t d = XAieTile_DmReadWord(&(TileInst[col][2]), 0x20 + (i*4));
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