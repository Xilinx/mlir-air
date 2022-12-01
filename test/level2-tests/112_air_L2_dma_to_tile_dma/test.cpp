//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <climits>
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

  for (int i = 0; i < 16; i++) {
    mlir_aie_write_buffer_buf0(xaie, i, 0xdecaf);
    mlir_aie_write_buffer_buf0(xaie, i + 16, 0xacafe);
  }

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
  for (int i = 0; i < 32; i++) {
    uint32_t toWrite = i + 0x0112;

    bank0_ptr[i] = toWrite + (1 << 28);
    bank1_ptr[i] = toWrite + (2 << 28);
  }

  // Read back the values from above
  for (int i = 0; i < 16; i++) {
    uint32_t word0 = bank0_ptr[i];
    uint32_t word1 = bank1_ptr[i];

    printf("%x %08X %08X\r\n", i, word0, word1);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  //
  // Set up a 4x4 herd starting 7,1
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 4, 1, 4);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  // globally bypass headers
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

  static l2_dma_cmd_t cmd;
  cmd.select = 7;
  cmd.length = 0;
  cmd.uram_addr = 1;
  cmd.id = 0;

  uint64_t stream = 0;
  air_packet_l2_dma(pkt, stream, cmd);

  //
  // send the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

  cmd.select = 2;
  cmd.length = 8;
  cmd.uram_addr = 0;
  cmd.id = 0x7;
  air_packet_l2_dma(pkt, stream, cmd);

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  mlir_aie_print_dma_status(xaie, 7, 4);

  uint32_t errs = 0;
  for (int i = 0; i < 32; i++) {
    uint32_t d;
    d = mlir_aie_read_buffer_buf0(xaie, i);
    printf("%d: %08X\n", i, d);
    if ((d & 0x0fffffff) != (i + 0x0112)) {
      printf("Word %i : Expect %08X, got %08X\n", i, i + 0x0112, d);
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
