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
#include "acdc_queue.h"
#include "hsa_defs.h"

#define SHMEM_BASE 0x020100000000LL

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#define BRAM_ADDR 0x020100000000LL
#define DMA_COUNT 256

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie.cpp"

}

extern "C" {

void _mlir_ciface_air_shim_memcpy(uint32_t id, uint64_t x, uint64_t y, void* t, uint64_t offset, uint64_t length) {

  uint64_t col = 2;
  auto burstlen = 4;
  XAieDma_Shim ShimDmaInst1;
  XAieDma_ShimInitialize(&(TileInst[col][0]), &ShimDmaInst1);
  // use BRAM_ADDR + 0x4000 as the data address
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)BRAM_ADDR), LOW_ADDR((u64)(BRAM_ADDR + 0x4000)), sizeof(u32) * DMA_COUNT);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, 1);

  auto ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_S2MM0);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_S2MM0)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }
  
}

}

template<typename T, int N>
struct tensor_t {
  T *d;
  T *aligned;
  size_t offset;
  size_t shape[N];
  size_t stride[N];

  size_t index(size_t n, size_t channel, size_t row, size_t col) const {
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    size_t idx = n * height * width * channels + channel * height * width + row * width + col;
    if (idx >= shape[0]*shape[1]*shape[2]*shape[3]) {
      printf("warning\n");
      return 0;
    }
    return idx;
  }

  tensor_t() {
    d = aligned = nullptr;
    offset = 0;
    for (int i=0; i<N; i++)
      shape[i] = stride[i] = 0;
  }
};

int
main(int argc, char *argv[])
{
  uint64_t row = 0;
  uint64_t col = 7;

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

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();

  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = i+1;
    XAieTile_DmWriteWord(&(TileInst[col][2]), 0x1000+i*4, d);
  }


  uint32_t *bram_ptr;

  // use BRAM_ADDR + 0x4000 as the data address
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)(0x4000 + (size_t)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR));
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = 0xdeadbeef;
      //printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }
  }

  mlir_start_cores();

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(herd_pkt);
  herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  // Set up a 1x3 herd starting 7,0
  herd_pkt->arg[0]  = AIR_PKT_TYPE_HERD_INITIALIZE;
  herd_pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  herd_pkt->arg[0] |= (1L << 40);
  herd_pkt->arg[0] |= (7L << 32);
  herd_pkt->arg[0] |= (3L << 24);
  herd_pkt->arg[0] |= (0L << 16);
  
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
    printf("%x\n", herd_pkt->completion_signal);
  }

  printf("loading aie_ctrl.so\n");
  auto handle = dlopen("./aie_ctrl.so", RTLD_NOW);
  if (!handle) {
    printf("%s\n",dlerror());
  }
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*, void*))dlsym(handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> input;
  input.shape[0] = 256;
  input.d = (uint32_t*)malloc(sizeof(uint32_t)*256);
  for (int i=0; i<input.shape[0]; i++) {
    input.d[i] = i;
  }

  auto i = &input;
  graph_fn(i, i);

  // wr_idx = queue_add_write_index(q, 1);
  // packet_id = wr_idx % q->size;

  // dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  // initialize_packet(pkt);
  // pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  // pkt->arg[0] = AIR_PKT_TYPE_SHIM_DMA_MEMCPY;
  // pkt->arg[0] |= (row << 16);
  // pkt->arg[0] |= (col << 32);
  // uint64_t flags = 0x1;
  // pkt->arg[0] |= (flags << 48);
  
  // uint32_t burst_len = 4;
  // uint64_t direction = 0;
  // uint64_t channel = XAIEDMA_SHIM_CHNUM_S2MM0;

  // pkt->arg[1] = burst_len;
  // pkt->arg[1] |= (direction << 32);
  // pkt->arg[1] |= (channel << 48);
  // pkt->arg[2] = BRAM_ADDR+0x4000;
  // pkt->arg[3] = DMA_COUNT*sizeof(float);

  // signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  // signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  // signal_store_release((signal_t*)&q->doorbell, wr_idx);

  // // We wrote data, so lets toggle locks 0 and 1
  // XAieTile_LockRelease(&(TileInst[col][2]), 0, 0x1, 0);
  // XAieTile_LockRelease(&(TileInst[col][2]), 1, 0x1, 0);

  // while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
  //   printf("packet completion signal timeout!\n");
  //   printf("%x\n", pkt->header);
  //   printf("%x\n", pkt->type);
  //   printf("%x\n", pkt->completion_signal);
  //   break;
  // }

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[i];
    if (d != (i+1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, i);
    }
  }

  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT-errors), DMA_COUNT);
  }

}
