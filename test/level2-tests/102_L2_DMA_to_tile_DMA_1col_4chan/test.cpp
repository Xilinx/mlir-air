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

  for (int i = 0; i < 16; i++) {
    mlir_aie_write_buffer_buf1(xaie, i, 0x9a01);
    mlir_aie_write_buffer_buf2(xaie, i, 0xba01);
    mlir_aie_write_buffer_buf4(xaie, i, 0x9a02);
    mlir_aie_write_buffer_buf5(xaie, i, 0xba02);
    mlir_aie_write_buffer_buf7(xaie, i, 0x9401);
    mlir_aie_write_buffer_buf8(xaie, i, 0xb401);
    mlir_aie_write_buffer_buf10(xaie, i, 0x9402);
    mlir_aie_write_buffer_buf11(xaie, i, 0xb402);
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
  for (int i = 0; i < 16; i++) {
    uint32_t toWrite = i;
    bank1_ptr[i] = toWrite + (2 << 28);
    bank0_ptr[i] = toWrite + (1 << 28);
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
  // Set up a 1x4 herd starting 7,1
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 1, 1, 4);
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

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  //
  // send the data
  //

  for (int sel = 0; sel < 4; sel++) {
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

    cmd.select = sel;
    cmd.length = (sel > 1) ? 8 : 4;
    cmd.uram_addr = 0;
    cmd.id = sel + 1;

    air_packet_l2_dma(pkt, stream, cmd);

    if (sel == 3) {
      air_queue_dispatch_and_wait(q, wr_idx, pkt);
    }
  }

  sleep(1);

  mlir_aie_print_dma_status(xaie, 7, 1);
  mlir_aie_print_dma_status(xaie, 7, 2);
  mlir_aie_print_dma_status(xaie, 7, 3);
  mlir_aie_print_dma_status(xaie, 7, 4);

  uint32_t errs = 0;
  for (int i = 0; i < 16; i++) {
    uint32_t check = i;
    uint32_t d;
    d = mlir_aie_read_buffer_buf1(xaie, i);
    if ((d & 0x0fffffff) != check) {
      printf("Part 0 : Word %i : Expect %x, got %08X\n", i, check, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t check = i;
    uint32_t d;
    d = mlir_aie_read_buffer_buf4(xaie, i);
    if ((d & 0x0fffffff) != check) {
      printf("Part 1 : Word %i : Expect %x, got %08X\n", i, check, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8);
    d = mlir_aie_read_buffer_buf7(xaie, i);
    printf("%d: %08x\n", i, d);
    if ((d & 0x0fffffff) != (c)) {
      printf("Prim 0 : Word %i : Expect %d, got %08X\n", i, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8) + 8;
    d = mlir_aie_read_buffer_buf8(xaie, i);
    printf("%d: %08x\n", i, d);
    if ((d & 0x0fffffff) != (c)) {
      printf("Prim 0 : Word %i : Expect %d, got %08X\n", i + 16, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8);
    d = mlir_aie_read_buffer_buf10(xaie, i);
    printf("%d: %08x\n", i, d);
    if ((d & 0x0fffffff) != (c)) {
      printf("Prim 1 : Word %i : Expect %d, got %08X\n", i, c, d);
      errs++;
    }
  }
  for (int i = 0; i < 16; i++) {
    uint32_t c, d;
    c = i % 4 + 4 * (i / 8) + 8;
    d = mlir_aie_read_buffer_buf11(xaie, i);
    printf("%d: %08x\n", i, d);
    if ((d & 0x0fffffff) != (c)) {
      printf("Prim 1 : Word %i : Expect %d, got %08X\n", i + 16, c, d);
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
