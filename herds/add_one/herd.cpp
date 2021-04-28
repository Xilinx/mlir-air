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
#include <dlfcn.h>

#include <xaiengine.h>

#include "air_host.h"
#include "air_tensor.h"
#include "hsa_defs.h"
#include "test_library.h"

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

#include "aie_inc.cpp"

//
// global q ptr
//
queue_t *q = nullptr;
uint32_t *bram_ptr;

}

#define BRAM_ADDR 0x4000+0x020100000000LL
#define DMA_COUNT 64

extern "C" {

long TILE_TO_SHIM_DMA[2][2][2] {0};
long ARG_TO_SHIM_DMA_CHANNEL[2][2][2] {0};
// Let's imagine the compiler made this function
void build_tile_to_shim_dma_mapping() {
  TILE_TO_SHIM_DMA[0][0][0] = 2L;
  TILE_TO_SHIM_DMA[1][0][0] = 2L;
  TILE_TO_SHIM_DMA[0][1][0] = 2L;
  TILE_TO_SHIM_DMA[1][1][0] = 2L;
  TILE_TO_SHIM_DMA[0][0][1] = 3L;
  TILE_TO_SHIM_DMA[1][0][1] = 3L;
  TILE_TO_SHIM_DMA[0][1][1] = 3L;
  TILE_TO_SHIM_DMA[1][1][1] = 3L;
}

void build_arg_to_shim_dma_channel_mapping() {
  ARG_TO_SHIM_DMA_CHANNEL[0][0][0] = XAIEDMA_SHIM_CHNUM_MM2S0;
  ARG_TO_SHIM_DMA_CHANNEL[1][0][0] = XAIEDMA_SHIM_CHNUM_S2MM0;
  ARG_TO_SHIM_DMA_CHANNEL[0][1][0] = XAIEDMA_SHIM_CHNUM_MM2S1;
  ARG_TO_SHIM_DMA_CHANNEL[1][1][0] = XAIEDMA_SHIM_CHNUM_S2MM1;
  ARG_TO_SHIM_DMA_CHANNEL[0][0][1] = XAIEDMA_SHIM_CHNUM_MM2S0;
  ARG_TO_SHIM_DMA_CHANNEL[1][0][1] = XAIEDMA_SHIM_CHNUM_S2MM0;
  ARG_TO_SHIM_DMA_CHANNEL[0][1][1] = XAIEDMA_SHIM_CHNUM_MM2S1;
  ARG_TO_SHIM_DMA_CHANNEL[1][1][1] = XAIEDMA_SHIM_CHNUM_S2MM1;
}

void _mlir_ciface_air_shim_memcpy(uint32_t id, uint64_t x, uint64_t y, void* t, uint64_t offset, uint64_t length) {
  printf("Do transfer with id %ld of length %ld on behalf of x=%ld, y=%ld using shim DMA %ld channel %ld, offset is %ld\n",
         id, length, x, y, TILE_TO_SHIM_DMA[id-1][x][y], ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y], offset);

  tensor_t<int32_t,1> *tt = (tensor_t<int32_t,1> *)t;

  // Used to use BRAM_ADDR + 0x4000 as the data address
  uint64_t addr = (u64)BRAM_ADDR;
  int32_t *bounce_buffer = (int32_t *)bram_ptr;
  if (id == 1) {
    // This is the input, so we need to take what is in t and put it into the BRAM
    for (int i=0; i<length; i++) {
      bounce_buffer[i] = tt->d[offset + i];
    }
  }

  ACDC_print_dma_status(TileInst[x+7][y+2]);
  ACDC_print_tile_status(TileInst[x+7][y+2]);

  auto burstlen = 4;
  XAieDma_Shim ShimDmaInst1;
  XAieDma_ShimInitialize(&(TileInst[TILE_TO_SHIM_DMA[id-1][x][y]][0]), &ShimDmaInst1);
  u8 bd = 1+ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y];
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, bd, HIGH_ADDR(addr), LOW_ADDR(addr), length*sizeof(uint32_t));
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, bd , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, bd);             // We don't really care, we just want these to be unique and none zero
  XAieDma_ShimSetStartBd((&ShimDmaInst1), ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y], bd);

  auto ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y]);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  XAieDma_ShimChControl((&ShimDmaInst1), ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y], XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y])) {
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }
  if (id == 2) {
    // This is the output, so we need to take what is in t and put it into the BRAM
    printf("Copy %ld samples to the output starting at %ld\n",length, offset);
    for (int i=0; i<length; i++) {
      tt->d[offset + i] = bounce_buffer[i];
    }
  }

  ACDC_print_dma_status(TileInst[x+7][y+2]);
  ACDC_print_tile_status(TileInst[x+7][y+2]);
}

}

int
main(int argc, char *argv[])
{
  uint64_t col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  ACDC_print_tile_status(TileInst[7][2]);

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  build_tile_to_shim_dma_mapping();
  build_arg_to_shim_dma_channel_mapping();

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<32*32; i++) {
      bram_ptr[i] = 0xdeface;
    }
  }

  // Stomp
  for (int i=0; i<DMA_COUNT; i++) {
    mlir_write_buffer_buf0(i, 0x0decaf);
    mlir_write_buffer_buf1(i, 0x1decaf);
    mlir_write_buffer_buf2(i, 0x2decaf);
    mlir_write_buffer_buf3(i, 0x3decaf);
    mlir_write_buffer_buf4(i, 0x4decaf);
    mlir_write_buffer_buf5(i, 0x5decaf);
    mlir_write_buffer_buf6(i, 0x6decaf);
    mlir_write_buffer_buf7(i, 0x7decaf);
  }

  ACDC_print_tile_status(TileInst[7][2]);

  // create the queue
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 2, /*row=*/2, 2);
  //air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  tensor_t<int32_t,1> input;
  tensor_t<int32_t,1> output;

  input.shape[0] = DMA_COUNT*4;
  input.d = input.aligned = (int32_t*)malloc(sizeof(int32_t)*input.shape[0]);

  output.shape[0] = DMA_COUNT*4;
  output.d = output.aligned = (int32_t*)malloc(sizeof(int32_t)*output.shape[0]);
  
  printf("loading aie_ctrl.so\n");
  auto handle = dlopen("./aie_ctrl.so", RTLD_NOW);
  if (!handle) {
    printf("%s\n",dlerror());
  }
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void*))dlsym(handle, "_mlir_ciface_myAddOne");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in add_one.so");

  for (int i=0; i<input.shape[0]; i++) {
    input.d[i] = i;
  }
  for (int i=0; i<16; i++) { 
    int32_t rb0 = mlir_read_buffer_buf0(i);
    int32_t rb1 = mlir_read_buffer_buf1(i);
    int32_t rb2 = mlir_read_buffer_buf2(i);
    int32_t rb3 = mlir_read_buffer_buf3(i);
    int32_t rb4 = mlir_read_buffer_buf4(i);
    int32_t rb5 = mlir_read_buffer_buf5(i);
    int32_t rb6 = mlir_read_buffer_buf6(i);
    int32_t rb7 = mlir_read_buffer_buf7(i);
    printf("before %d [7][2] : %08X -> %08X, [8][2] :%08X -> %08X, [7][3] : %08X -> %08X, [8][3] :%08X -> %08X\n", i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7);
  }

  void *i,*o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  for (int i=0; i<16; i++) { 
    int32_t rb0 = mlir_read_buffer_buf0(i);
    int32_t rb1 = mlir_read_buffer_buf1(i);
    int32_t rb2 = mlir_read_buffer_buf2(i);
    int32_t rb3 = mlir_read_buffer_buf3(i);
    int32_t rb4 = mlir_read_buffer_buf4(i);
    int32_t rb5 = mlir_read_buffer_buf5(i);
    int32_t rb6 = mlir_read_buffer_buf6(i);
    int32_t rb7 = mlir_read_buffer_buf7(i);
    printf(" after %d [7][2] : %08X -> %08X, [8][2] :%08X -> %08X, [7][3] : %08X -> %08X, [8][3] :%08X -> %08X\n", i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7);
  }

  ACDC_print_tile_status(TileInst[7][2]);

  int errors = 0;
  for (int i=0; i<output.shape[0]; i++) {
    uint32_t d = output.d[i];
    if (d != (i+1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, i);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %ld/%ld.\n", (output.shape[0]-errors), output.shape[0]);
  }
}
