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

#include "air_host.h"
#include "test_library.h"

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie.inc"
#undef TileInst
#undef TileDMAInst

}

int
main(int argc, char *argv[])
{
  uint64_t col = 7;
  uint64_t row = 0;

  xaie = air_init_libxaie1();

  ACDC_print_dma_status(xaie->TileInst[7][2]);

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  uint32_t *bram_ptr;

  #define DMA_COUNT 512

  // Ascending plus 2 sequence in the tile memory, and toggle the associated lock
  for (int i=0; i<DMA_COUNT; i++) {
    if (i<(DMA_COUNT/2))
      mlir_write_buffer_a(i, i+2);
    else
      mlir_write_buffer_b(i-(DMA_COUNT/2), i+2);
  }

  XAieTile_LockRelease(&(xaie->TileInst[7][2]), 0, 0x1, 0);
  XAieTile_LockRelease(&(xaie->TileInst[7][2]), 1, 0x1, 0);

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // Let's make a buffer that we can transfer in the same BRAM, after the queue of HSA packets
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE+0x4000);
  // Lets stomp over it!
  for (int i=0;i<DMA_COUNT;i++) {
    bram_ptr[i] = 0xdeadbeef;
  }


  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  dispatch_packet_t *dev_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_device_init(dev_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(q, wr_idx, dev_pkt);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  dispatch_packet_t *cpypkt0 = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(cpypkt0, 0, col, 0, 0, 8, 2, AIR_VCK190_SHMEM_BASE+0x4000, DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, cpypkt0);

  ACDC_print_dma_status(xaie->TileInst[7][2]);

  uint32_t errs = 0;
  // Let go check the tile memory
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d;
    if (i<(DMA_COUNT/2))
      d = mlir_read_buffer_a(i);
    else
      d = mlir_read_buffer_b(i-(DMA_COUNT/2));

    if (d != i+2) {
      printf("ERROR: Tile Memory id %d Expected %08X, got %08X\n", i, i+2, d);
      errs++;
    }
  }
  for (int i=0; i<DMA_COUNT; i++) {
    if (bram_ptr[i] != 2+i) {
      printf("ERROR: L2 Memory id %d Expected %08X, got %08X\n", i, i+2, bram_ptr[i]);
      errs++;
    }
  }

  if (errs == 0)
    printf("PASS!\n");
  else
    printf("fail %d/%d.\n",DMA_COUNT-errs, DMA_COUNT);


  for (int bd=0;bd<16;bd++) {
    // Take no prisoners.  No regerts
    // Overwrites the DMA_BDX_Control registers
    XAieGbl_Write32(xaie->TileInst[7][0].TileAddr + 0x0001D008+(bd*0x14), 0x0);
    XAieGbl_Write32(xaie->TileInst[7][2].TileAddr + 0x0001D018+(bd*0x20), 0x0);
  }
}
