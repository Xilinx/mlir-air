//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "air_host.h"
#include "test_library.h"

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  for (int i = 0; i < 512; i++) {
    mlir_aie_write_buffer_buf1(xaie, i, i + 0x1000);
    mlir_aie_write_buffer_buf2(xaie, i, i + 0x2000);
    mlir_aie_write_buffer_buf3(xaie, i, i + 0x3000);
    mlir_aie_write_buffer_buf4(xaie, i, i + 0x4000);
  }

  mlir_aie_print_dma_status(xaie, 7, 1);
  mlir_aie_print_dma_status(xaie, 7, 2);
  mlir_aie_print_dma_status(xaie, 7, 3);
  mlir_aie_print_dma_status(xaie, 7, 4);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE);
  uint32_t *bank1_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0x20000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it
  // goes in
  for (int i = 0; i < 512 * 5; i++) {
    uint32_t offset = i;
    uint32_t toWrite = i;

    bank1_ptr[offset] = toWrite + (2 << 28);
    bank0_ptr[offset] = toWrite + (1 << 28);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  //
  // Set up a 1x4 herd starting 7,1
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 1, 1, 4);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  //
  // enable headers
  //
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

  static l2_dma_cmd_t cmd;
  cmd.select = 7;
  cmd.length = 0;
  cmd.uram_addr = 0;
  cmd.id = 0;

  air_packet_l2_dma(pkt, 0, cmd);

  //
  // send the data
  //
  int sel = 2;
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

  cmd.select = sel;
  cmd.length = 128;
  cmd.uram_addr = 0;
  cmd.id = sel + 1;

  air_packet_l2_dma(pkt, 0, cmd);

  //
  // read the data back
  //
  sel = 4;
  for (int i = 0; i < 4; i++) {
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

    cmd.select = sel;
    cmd.length = 128;
    cmd.uram_addr = 128 + 128 * i;
    cmd.id = 0xA + i;

    air_packet_l2_dma(pkt, 0, cmd);
  }
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  sleep(1);

  mlir_aie_print_dma_status(xaie, 7, 1);
  mlir_aie_print_dma_status(xaie, 7, 2);
  mlir_aie_print_dma_status(xaie, 7, 3);
  mlir_aie_print_dma_status(xaie, 7, 4);

  uint32_t errs = 0;
  int it = 0;
  for (int i = 512; i < 2560; i++) {
    if (!(i % 512))
      it++;
    uint32_t d0;
    d0 = bank0_ptr[i - 512 * it];
    uint32_t d;
    d = bank0_ptr[i];
    if (d != d0) {
      printf("Part 0 %i : Expect %08X, got %08X\n", i, d0, d);
      errs++;
    }
  }

  if (errs) {
    printf("FAIL: %d errors\n", errs);
    return -1;
  } else {
    printf("PASS!\n");
    return 0;
  }
}
