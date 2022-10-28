//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
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

  for (int i = 0; i < 32; i++) {
    mlir_aie_write_buffer_a(xaie, i, 0xcafe01);
    mlir_aie_write_buffer_b(xaie, i, 0xcafe02);
    mlir_aie_write_buffer_c(xaie, i, 0xcafe03);
    mlir_aie_write_buffer_d(xaie, i, 0xcafe04);
    mlir_aie_write_buffer_e(xaie, i, 0xcafe05);
    mlir_aie_write_buffer_f(xaie, i, 0xcafe06);
    mlir_aie_write_buffer_g(xaie, i, 0xcafe07);
    mlir_aie_write_buffer_i(xaie, i, 0xcafe08);
  }

  mlir_aie_print_dma_status(xaie, 7, 2);
  mlir_aie_print_dma_status(xaie, 8, 2);
  mlir_aie_print_dma_status(xaie, 9, 2);
  mlir_aie_print_dma_status(xaie, 10, 2);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0 * 0x20000);
  uint32_t *bank1_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 1 * 0x20000);
  uint32_t *bank2_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 2 * 0x20000);
  uint32_t *bank3_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 3 * 0x20000);
  uint32_t *bank4_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 4 * 0x20000);
  uint32_t *bank5_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 5 * 0x20000);
  uint32_t *bank6_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 6 * 0x20000);
  uint32_t *bank7_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 7 * 0x20000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 1 for the upper memory as it
  // goes in
  for (int i = 0; i < 16; i++) {
    uint32_t toWrite = i;
    toWrite += (0x100000);
    bank1_ptr[i] = toWrite + (2 << 28);
    bank0_ptr[i] = toWrite + (1 << 28);
    toWrite += (0x200000);
    bank3_ptr[i] = toWrite + (2 << 28);
    bank2_ptr[i] = toWrite + (1 << 28);
    toWrite += (0x400000);
    bank5_ptr[i] = toWrite + (2 << 28);
    bank4_ptr[i] = toWrite + (1 << 28);
    toWrite += (0x800000);
    bank7_ptr[i] = toWrite + (2 << 28);
    bank6_ptr[i] = toWrite + (1 << 28);
  }

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

  for (int stream = 0; stream < 4; stream++) {
    // globally bypass headers
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

  //
  // send the data
  //

  for (int stream = 0; stream < 4; stream++) {
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

    static l2_dma_cmd_t cmd;
    cmd.select = 0;
    cmd.length = 4;
    cmd.uram_addr = 0;
    cmd.id = stream;

    air_packet_l2_dma(pkt, stream, cmd);

    signal_create(1, 0, NULL, (signal_t *)&pkt->completion_signal);
    if (stream == 3) {
      air_queue_dispatch_and_wait(q, wr_idx, pkt);
    }
  }

  sleep(1);
  mlir_aie_print_dma_status(xaie, 7, 2);
  mlir_aie_print_dma_status(xaie, 8, 2);
  mlir_aie_print_dma_status(xaie, 9, 2);
  mlir_aie_print_dma_status(xaie, 10, 2);

  printf("\nChecking the output...\n");

  uint32_t errs = 0;
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8);
    d = mlir_aie_read_buffer_a(xaie, i) - 0x100000;
    if ((d & 0xffff) != (c)) {
      printf("[7] Word %i : expect %d, got %08x\n", i, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8) + 8;
    d = mlir_aie_read_buffer_b(xaie, i) - 0x100000;
    if ((d & 0xffff) != (c)) {
      printf("[7] Word %i : expect %d, got %08x\n", i + 16, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8);
    d = mlir_aie_read_buffer_a(xaie, i) - 0x300000;
    if ((d & 0xffff) != (c)) {
      printf("[8] Word %i : expect %d, got %08x\n", i, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8) + 8;
    d = mlir_aie_read_buffer_b(xaie, i) - 0x300000;
    if ((d & 0xffff) != (c)) {
      printf("[8] Word %i : expect %d, got %08x\n", i + 16, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8);
    d = mlir_aie_read_buffer_a(xaie, i) - 0x700000;
    if ((d & 0xffff) != (c)) {
      printf("[9] Word %i : expect %d, got %08x\n", i, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8) + 8;
    d = mlir_aie_read_buffer_b(xaie, i) - 0x700000;
    if ((d & 0xffff) != (c)) {
      printf("[9] Word %i : expect %d, got %08x\n", i + 16, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8);
    d = mlir_aie_read_buffer_a(xaie, i) - 0xf00000;
    if ((d & 0xffff) != (c)) {
      printf("[A] Word %i : expect %d, got %08x\n", i, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8) + 8;
    d = mlir_aie_read_buffer_b(xaie, i) - 0xf00000;
    if ((d & 0xffff) != (c)) {
      printf("[A] Word %i : expect %d, got %08x\n", i + 16, c, d);
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
