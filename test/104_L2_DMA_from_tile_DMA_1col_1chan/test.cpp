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

#include "aie_inc.cpp"

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

  XAieTile_LockRelease(&(xaie->TileInst[7][4]), 2, 1, 0);
  auto lock_ret2 = XAieTile_LockAcquire(&(xaie->TileInst[7][4]), 2, 1, 10000);
  assert(lock_ret2);

  for (int i=0; i<16; i++) {
    mlir_aie_write_buffer_buf1(xaie, i,i+0x1000);
  }

  mlir_aie_print_dma_status(xaie, 7, 4);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE);
  uint32_t *bank1_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, L2_DMA_BASE+0x20000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it goes in
  for (int i=0;i<32;i++) {
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
    if (upper_lower)
      bank1_ptr[offset] = toWrite;
    else
      bank0_ptr[offset] = toWrite;

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
  // Set up a 1x1 herd starting 7,4
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 1, 4, 1);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  // globally bypass headers
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

  uint64_t stream = 0;
  pkt->arg[1] = stream;
  pkt->arg[2] = 0;
  pkt->arg[2] |= ((uint64_t)cmd.select) << 32;
  pkt->arg[2] |= cmd.length << 18;
  pkt->arg[2] |= cmd.uram_addr << 5;
  pkt->arg[2] |= cmd.id;

  air_queue_dispatch_and_wait(q, wr_idx, pkt);


  // release the lock on the tile DMA
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  // lock packet
  uint32_t herd_id = 0;
  uint32_t lock_id = 1;
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_aie_lock(lock_pkt, herd_id, lock_id, /*acq_rel*/1, /*value*/1, 0, 0);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  pkt->arg[0] = AIR_PKT_TYPE_PUT_STREAM;

  cmd.select = 4;
  cmd.length = 4;
  cmd.uram_addr = 0;
  cmd.id = 0xa;

  pkt->arg[1] = stream;
  pkt->arg[2] = 0;
  pkt->arg[2] |= ((uint64_t)cmd.select) << 32;
  pkt->arg[2] |= cmd.length << 18;
  pkt->arg[2] |= cmd.uram_addr << 5;
  pkt->arg[2] |= cmd.id;

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  mlir_aie_print_dma_status(xaie, 7, 4);
  
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
    d = bank0_ptr[offset];
    if ((d & 0x0fffffff) != (i+0x1000)) {
      printf("Word %i : Expect %d, got %08X\n",i, i, d);
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
