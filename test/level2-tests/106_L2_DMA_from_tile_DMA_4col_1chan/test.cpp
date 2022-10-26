//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "air_host.h"

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  aie_libxaie_ctx_t *xaie = air_init_libxaie();

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  mlir_aie_release_lock(xaie, 7, 4, 1, 0, 0);
  auto lock_ret = mlir_aie_acquire_lock(xaie, 7, 4, 1, 0, 1000);
  assert(lock_ret);

  mlir_aie_release_lock(xaie, 8, 4, 1, 0, 0);
  auto lock_ret2 = mlir_aie_acquire_lock(xaie, 8, 4, 1, 0, 1000);
  assert(lock_ret2);

  mlir_aie_release_lock(xaie, 9, 4, 1, 0, 0);
  auto lock_ret3 = mlir_aie_acquire_lock(xaie, 9, 4, 1, 0, 1000);
  assert(lock_ret3);

  mlir_aie_release_lock(xaie, 10, 4, 1, 0, 0);
  auto lock_ret4 = mlir_aie_acquire_lock(xaie, 10, 4, 1, 0, 1000);
  assert(lock_ret4);

  for (int i = 0; i < 16; i++) {
    mlir_aie_write_buffer_buf1(xaie, i, i + 0x1000);
    mlir_aie_write_buffer_buf2(xaie, i, i + 0x2000);
    mlir_aie_write_buffer_buf3(xaie, i, i + 0x3000);
    mlir_aie_write_buffer_buf4(xaie, i, i + 0x4000);
  }

  mlir_aie_print_dma_status(xaie, 7, 4);
  mlir_aie_print_dma_status(xaie, 8, 4);
  mlir_aie_print_dma_status(xaie, 9, 4);
  mlir_aie_print_dma_status(xaie, 10, 4);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_A_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE);
  uint32_t *bank1_A_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0x20000);
  uint32_t *bank0_B_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0x40000);
  uint32_t *bank1_B_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0x60000);
  uint32_t *bank0_C_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0x80000);
  uint32_t *bank1_C_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0xA0000);
  uint32_t *bank0_D_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0xC0000);
  uint32_t *bank1_D_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0xE0000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it
  // goes in
  for (int i = 0; i < 16; i++) {
    uint32_t toWrite = 0xcafe00 + i;
    bank1_A_ptr[i] = toWrite + (2 << 28);
    bank1_B_ptr[i] = toWrite + (2 << 28);
    bank1_C_ptr[i] = toWrite + (2 << 28);
    bank1_D_ptr[i] = toWrite + (2 << 28);
    bank0_A_ptr[i] = toWrite + (1 << 28);
    bank0_B_ptr[i] = toWrite + (1 << 28);
    bank0_C_ptr[i] = toWrite + (1 << 28);
    bank0_D_ptr[i] = toWrite + (1 << 28);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  //
  // Set up a 4x1 herd starting 7,4
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 4, 4, 1);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  // globally bypass headers
  for (uint64_t stream = 0; stream < 4; stream++) {
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

    static l2_dma_cmd_t cmd;
    cmd.select = 7;
    cmd.length = 0;
    cmd.uram_addr = 1;
    cmd.id = 0;

    air_packet_l2_dma(pkt, stream, cmd);

    air_queue_dispatch_and_wait(q, wr_idx, pkt);
  }

  // release the lock on the tile DMA
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  // lock packet
  uint32_t herd_id = 0;
  uint32_t lock_id = 1;
  dispatch_packet_t *lock_pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_aie_lock_range(lock_pkt, herd_id, lock_id, /*acq_rel*/ 1,
                            /*value*/ 1, 0, 4, 0, 1);

  //
  // read the data
  //

  for (uint64_t stream = 0; stream < 4; stream++) {
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

    static l2_dma_cmd_t cmd;
    cmd.select = 4;
    cmd.length = 4;
    cmd.uram_addr = 0;
    cmd.id = 0x2 + stream;

    air_packet_l2_dma(pkt, stream, cmd);
  }

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  // sleep(1);
  mlir_aie_print_dma_status(xaie, 7, 4);
  mlir_aie_print_dma_status(xaie, 8, 4);
  mlir_aie_print_dma_status(xaie, 9, 4);
  mlir_aie_print_dma_status(xaie, 10, 4);

  uint32_t errs = 0;
  for (int i = 0; i < 16; i++) {
    uint32_t d;
    d = bank0_A_ptr[i];
    if ((d & 0x0fffffff) != (i + 0x1000)) {
      printf("Part 0 A %i : Expect %d, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t d;
    d = bank0_B_ptr[i];
    if ((d & 0x0fffffff) != (i + 0x2000)) {
      printf("Part 0 B %i : Expect %d, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t d;
    d = bank0_C_ptr[i];
    if ((d & 0x0fffffff) != (i + 0x3000)) {
      printf("Part 0 C %i : Expect %d, got %08X\n", i, i, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t d;
    d = bank0_D_ptr[i];
    if ((d & 0x0fffffff) != (i + 0x4000)) {
      printf("Part 0 D %i : Expect %d, got %08X\n", i, i, d);
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
