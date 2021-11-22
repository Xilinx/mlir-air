// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cstdio>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "air_host.h"
#include "test_library.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie_inc.cpp"
#undef TileInst
#undef TileDMAInst

}

#define L2_DMA_BASE 0x020240000000LL
#define SHMEM_BASE  0x020100000000LL

struct dma_cmd_t {
  uint8_t select;
  uint16_t length;
  uint16_t uram_addr;
  uint8_t id;
};

struct dma_rsp_t {
	uint8_t id;
};

int main(int argc, char *argv[])
{

  xaie = air_init_libxaie1();
  
  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  XAieTile_LockRelease(&(xaie->TileInst[7][4]), 1, 0, 0);
  auto lock_ret = XAieTile_LockAcquire(&(xaie->TileInst[7][4]), 1, 0, 10000);
  assert(lock_ret);

  XAieTile_LockRelease(&(xaie->TileInst[8][4]), 1, 0, 0);
  auto lock_ret2 = XAieTile_LockAcquire(&(xaie->TileInst[8][4]), 1, 0, 10000);
  assert(lock_ret2);

  XAieTile_LockRelease(&(xaie->TileInst[9][4]), 1, 0, 0);
  auto lock_ret3 = XAieTile_LockAcquire(&(xaie->TileInst[9][4]), 1, 0, 10000);
  assert(lock_ret3);

  XAieTile_LockRelease(&(xaie->TileInst[10][4]), 1, 0, 0);
  auto lock_ret4 = XAieTile_LockAcquire(&(xaie->TileInst[10][4]), 1, 0, 10000);
  assert(lock_ret4);

  for (int i=0; i<16; i++) {
    mlir_write_buffer_buf1(i,i+0x1000);
    mlir_write_buffer_buf2(i,i+0x2000);
    mlir_write_buffer_buf3(i,i+0x3000);
    mlir_write_buffer_buf4(i,i+0x4000);
  }

  ACDC_print_dma_status(xaie->TileInst[7][4]);
  ACDC_print_dma_status(xaie->TileInst[8][4]);
  ACDC_print_dma_status(xaie->TileInst[9][4]);
  ACDC_print_dma_status(xaie->TileInst[10][4]);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_A_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE);
  uint32_t *bank1_A_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0x20000);
  uint32_t *bank0_B_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0x40000);
  uint32_t *bank1_B_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0x60000);
  uint32_t *bank0_C_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0x80000);
  uint32_t *bank1_C_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0xA0000);
  uint32_t *bank0_D_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0xC0000);
  uint32_t *bank1_D_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0xE0000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it goes in
  for (int i=0;i<16;i++) {
    uint32_t upper_lower = (i%8)/4;
    uint32_t first128_second128 = i%2;
    uint32_t first64_second64 = (i%16)/8;
    uint32_t first32_second32 = (i/2)%2;
    uint32_t offset = (first128_second128)*4;
    offset += (first64_second64)*2;
    offset += first32_second32;
    offset += (i/16)*8;
    uint32_t toWrite = 0xcafe00 + i + (((upper_lower)+1) << 28);

    printf("%d : %d %d %d %d %d %08X\n",i,upper_lower, first128_second128, first64_second64, first32_second32, offset, toWrite);
    if (upper_lower) {
      bank1_A_ptr[offset] = toWrite;
      bank1_B_ptr[offset] = toWrite;
      bank1_C_ptr[offset] = toWrite;
      bank1_D_ptr[offset] = toWrite;
    } else {
      bank0_A_ptr[offset] = toWrite;
      bank0_B_ptr[offset] = toWrite;
      bank0_C_ptr[offset] = toWrite;
      bank0_D_ptr[offset] = toWrite;
    }

  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  //
  // Set up a 4x1 herd starting 7,4
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 4, 4, 1);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
    
  // globally bypass headers
  for (uint64_t stream=0; stream < 4; stream++) {
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;

    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    initialize_packet(pkt);
    pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
    pkt->arg[0] = AIR_PKT_TYPE_PUT_STREAM;

    static dma_cmd_t cmd;
    cmd.select = 7;
    cmd.length = 0;
    cmd.uram_addr = 1;
    cmd.id = 0;

    pkt->arg[1] = stream;
    pkt->arg[2] = 0;
    pkt->arg[2] |= ((uint64_t)cmd.select) << 32;
    pkt->arg[2] |= cmd.length << 18;
    pkt->arg[2] |= cmd.uram_addr << 5;
    pkt->arg[2] |= cmd.id;

    air_queue_dispatch_and_wait(q, wr_idx, pkt);
  }

  // release the lock on the tile DMA
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  // lock packet
  uint32_t herd_id = 0;
  uint32_t lock_id = 1;
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_aie_lock_range(lock_pkt, herd_id, lock_id, /*acq_rel*/1, /*value*/1, 0, 4, 0, 1);

  //
  // read the data
  //

  for (uint64_t stream=0; stream < 4; stream++) {
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;

    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    initialize_packet(pkt);
    pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
    pkt->arg[0] = AIR_PKT_TYPE_PUT_STREAM;

    static dma_cmd_t cmd;
    cmd.select = 4;
    cmd.length = 4;
    cmd.uram_addr = 0;
    cmd.id = 0x2+stream;

    pkt->arg[1] = stream;
    pkt->arg[2] = 0;
    pkt->arg[2] |= ((uint64_t)cmd.select) << 32;
    pkt->arg[2] |= cmd.length << 18;
    pkt->arg[2] |= cmd.uram_addr << 5;
    pkt->arg[2] |= cmd.id;
  }

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  //sleep(1);
  ACDC_print_dma_status(xaie->TileInst[7][4]);
  ACDC_print_dma_status(xaie->TileInst[8][4]);
  ACDC_print_dma_status(xaie->TileInst[9][4]);
  ACDC_print_dma_status(xaie->TileInst[10][4]);
  
  uint32_t errs = 0;
  for (int i=0; i<16; i++) {
    uint32_t upper_lower = i/4;
    uint32_t first128_second128 = i%2;
    uint32_t first64_second64 = (i%16)/8;
    uint32_t first32_second32 = (i/2)%2;
    uint32_t offset = (first128_second128)*4;
    offset += first64_second64*4;
    offset += first32_second32;
    offset += upper_lower*2;
    uint32_t d;
    d = bank0_A_ptr[offset];
    if ((d & 0x0fffffff) != (i+0x1000)) {
      printf("Part 0 A %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }
  for (int i=0; i<16; i++) {
    uint32_t upper_lower = i/4;
    uint32_t first128_second128 = i%2;
    uint32_t first64_second64 = (i%16)/8;
    uint32_t first32_second32 = (i/2)%2;
    uint32_t offset = (first128_second128)*4;
    offset += first64_second64*4;
    offset += first32_second32;
    offset += upper_lower*2;
    uint32_t d;
    d = bank0_B_ptr[offset];
    if ((d & 0x0fffffff) != (i+0x2000)) {
      printf("Part 0 B %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }
  for (int i=0; i<16; i++) {
    uint32_t upper_lower = i/4;
    uint32_t first128_second128 = i%2;
    uint32_t first64_second64 = (i%16)/8;
    uint32_t first32_second32 = (i/2)%2;
    uint32_t offset = (first128_second128)*4;
    offset += first64_second64*4;
    offset += first32_second32;
    offset += upper_lower*2;
    uint32_t d;
    d = bank0_C_ptr[offset];
    if ((d & 0x0fffffff) != (i+0x3000)) {
      printf("Part 0 C %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }
  for (int i=0; i<16; i++) {
    uint32_t upper_lower = i/4;
    uint32_t first128_second128 = i%2;
    uint32_t first64_second64 = (i%16)/8;
    uint32_t first32_second32 = (i/2)%2;
    uint32_t offset = (first128_second128)*4;
    offset += first64_second64*4;
    offset += first32_second32;
    offset += upper_lower*2;
    uint32_t d;
    d = bank0_D_ptr[offset];
    if ((d & 0x0fffffff) != (i+0x4000)) {
      printf("Part 0 D %i : Expect %d, got %08X\n",i, i, d);
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
