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

#include "aie.0.inc"

}

extern "C" {

void _mlir_ciface_air_shim_memcpy(uint32_t id, uint64_t x, uint64_t y, void* t, uint64_t offset, uint64_t length) {

  uint64_t col = 2;
  auto burstlen = 4;
  XAieDma_Shim ShimDmaInst1;
  XAieDma_ShimInitialize(&(TileInst[col][0]), &ShimDmaInst1);
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)BRAM_ADDR), LOW_ADDR((u64)BRAM_ADDR), sizeof(u32) * DMA_COUNT);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, 1);

  auto ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }
}

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

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(herd_pkt);
  herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  // Set up a 1x1 herd starting 7,2
  herd_pkt->arg[0]  = AIR_PKT_TYPE_HERD_INITIALIZE;
  herd_pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  herd_pkt->arg[0] |= (1L << 40);
  herd_pkt->arg[0] |= (7L << 32);
  herd_pkt->arg[0] |= (1L << 24);
  herd_pkt->arg[0] |= (2L << 16);
  
  herd_pkt->arg[1] = 0;  // Herd ID 0
  herd_pkt->arg[2] = 0;
  herd_pkt->arg[3] = 0;

  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  printf("loading aie_ctrl.so\n");
  auto handle = dlopen("./aie_ctrl.so", RTLD_NOW);
  if (!handle) {
    printf("%s\n",dlerror());
  }
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void*))dlsym(handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> input;6
  input.shape[0] = 256;
  input.d = (uint32_t*)malloc(sizeof(uint32_t)*256);
  for (int i=0; i<input.shape[0]; i++) {
    input.d[i] = i;
  }

  auto i = &input;
  graph_fn(i, i);

  printCoreStatus(7,2);
  // we copied the start of the shared bram into tile memory,
  // fish out the queue id and check it
  uint32_t d = mlir_read_buffer_buf0(24);
  printf("ID %x\n", d);

  if (d == 0xacdc)
    printf("PASS!\n");
  else
    printf("fail.\n");

}
