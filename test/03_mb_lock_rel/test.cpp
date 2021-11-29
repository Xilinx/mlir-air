// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <xaiengine.h>

#include "air_host.h"
#include "test_library.h"

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#include "aie_inc.cpp"

int main(int argc, char *argv[])
{
  auto row = 2;
  auto col = 7;
  auto num_rows = 1;
  auto num_cols = 1;

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);

  mlir_aie_print_tile_status(xaie, col, row);

  // Run auto generated config functions
  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);

  XAieTile_LockRelease(&(xaie->TileInst[col][2]), 0, 1, 0);
  auto lock_ret = XAieTile_LockAcquire(&(xaie->TileInst[col][2]), 0, 1, 10000);
  assert(lock_ret);

  mlir_aie_configure_dmas(xaie);

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // setup the shim dma descriptors
  XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

  #define DMA_COUNT 256

  auto burstlen = 4;
  XAieDma_ShimInitialize(&(xaie->TileInst[col][0]), &ShimDmaInst1);
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)AIR_VCK190_SHMEM_BASE), LOW_ADDR((u64)AIR_VCK190_SHMEM_BASE+0x1000), sizeof(u32) * DMA_COUNT);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, 1);

  auto cnt = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0);
  if (cnt)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, cnt);

  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  uint32_t herd_id = 0;
  uint32_t lock_id = 0;

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  // Set up the worlds smallest herd at 7,2
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, herd_id, col, num_cols, row, num_rows);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // reserve another packet in the queue
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  // lock packet
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_aie_lock(lock_pkt, herd_id, lock_id, /*acq_rel*/1, /*value*/0, 0, 0);
  air_queue_dispatch_and_wait(q, wr_idx, lock_pkt);

  //XAieTile_LockRelease(&(xaie->TileInst[col][2]), 0, 0, 0);

  // wait for shim dma to finish
  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S0)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  // we copied the start of the shared bram into tile memory,
  // fish out the queue id and check it
  uint32_t d = mlir_aie_read_buffer_b0(xaie, 24);
  printf("ID %x\n", d);

  if (d == 0xacdc) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail.\n");
    return -1;
  }
}
