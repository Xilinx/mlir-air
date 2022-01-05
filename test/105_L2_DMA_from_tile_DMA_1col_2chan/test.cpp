// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cstdio>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "air_host.h"

#include "aie_inc.cpp"

int main(int argc, char *argv[])
{
  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);
  
  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  XAieTile_LockRelease(&(xaie->TileInst[7][4]), 1, 0, 0);
  auto lock_ret = XAieTile_LockAcquire(&(xaie->TileInst[7][4]), 1, 0, 10000);
  assert(lock_ret);

  XAieTile_LockRelease(&(xaie->TileInst[7][3]), 1, 0, 0);
  auto lock_ret2 = XAieTile_LockAcquire(&(xaie->TileInst[7][3]), 1, 0, 10000);
  assert(lock_ret2);

  for (int i=0; i<16; i++) {
    mlir_aie_write_buffer_buf1(xaie, i,i+0x1000);
    mlir_aie_write_buffer_buf2(xaie, i,i+0x2000);
  }

  mlir_aie_print_dma_status(xaie, 7, 3);
  mlir_aie_print_dma_status(xaie, 7, 4);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE);
  uint32_t *bank1_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+0x20000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it goes in
  for (int i=0;i<16;i++) {
    uint32_t toWrite = 0xcafe00 + i;
    bank1_ptr[i] = toWrite + (2 << 28);
    bank0_ptr[i] = toWrite + (1 << 28);
  }

  // Read back the value above it

  for (int i=0;i<16;i++) {
    uint32_t word0 = bank0_ptr[i];
    uint32_t word1 = bank1_ptr[i];

    printf("%x %08X %08X\r\n", i, word0, word1);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  //
  // Set up a 1x2 herd starting 7,3
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 1, 3, 2);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
    
  // globally bypass headers
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

  static l2_dma_cmd_t cmd;
  cmd.select = 7;
  cmd.length = 0;
  cmd.uram_addr = 1;
  cmd.id = 0;

  uint64_t stream = 0;
  air_packet_l2_dma(pkt, stream, cmd);

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  // release the lock on the tile DMAs
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  // lock packet
  uint32_t herd_id = 0;
  uint32_t lock_id = 1;
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_aie_lock_range(lock_pkt, herd_id, lock_id, /*acq_rel*/1, /*value*/1, 0, 1, 0, 2);

  //
  // read the data
  //

  for (int sel = 4; sel < 6; sel++) { 
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

    cmd.select = sel;
    cmd.length = 4;
    cmd.uram_addr = 0;
    cmd.id = 0x2+sel;

    air_packet_l2_dma(pkt, stream, cmd);
  }

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  mlir_aie_print_dma_status(xaie, 7, 3);
  mlir_aie_print_dma_status(xaie, 7, 4);
  
  uint32_t errs = 0;
  for (int i=0; i<16; i++) {
    uint32_t d;
    d = bank0_ptr[i];
    if ((d & 0x0fffffff) != (i+0x1000)) {
      printf("Part 0 %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }
  for (int i=0; i<16; i++) {
    uint32_t d;
    d = bank1_ptr[i];
    if ((d & 0x0fffffff) != (i+0x2000)) {
      printf("Part 1 %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }

  if (errs) {
    printf("FAIL: %d errors\n", errs);
    return -1;
  }
  else {
    printf("PASS!\n");
    return 0;
  }
}
