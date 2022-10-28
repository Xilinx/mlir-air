//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
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
#include <climits>
#include <cstdio>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "air_host.h"

#include "aie_inc.cpp"

#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 4
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

int main(int argc, char *argv[]) {
  aie_libxaie_ctx_t *xaie = air_init_libxaie();

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  for (int i = 0; i < TILE_SIZE; i++) {
    mlir_aie_write_buffer_buf0(xaie, i, i + 0xacdc1000);
    mlir_aie_write_buffer_buf1(xaie, i, i + 0xacdc2000);
  }
  for (int i = 0; i < TILE_SIZE; i++) {
    uint32_t word0 = mlir_aie_read_buffer_buf0(xaie, i);
    uint32_t word1 = mlir_aie_read_buffer_buf1(xaie, i);

    printf("Tiles %x %08X %08X\r\n", i, word0, word1);
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
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t toWrite = i;

    bank0_ptr[i] = toWrite;
    bank1_ptr[i + IMAGE_SIZE] = 0xdeafcafe;
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

  //
  // Set up a 1x1 herd starting 7,4
  //
  air_packet_herd_init(pkt, 0, 7, 1, 4, 1);

  // dispatch packet
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  // globally bypass headers
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;

  l2_dma_cmd_t cmd;
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

  // Send the packet to write the tiles
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_a =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, 7, 1, 0, 4, /*space*/ 1, 0,
                       TILE_WIDTH * sizeof(float), TILE_HEIGHT,
                       IMAGE_WIDTH * sizeof(float), 1, 0, 1, 0);

  //
  // read the data back
  //

  // Send the packet to read from the tiles
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_c =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_c, 0, 7, 0, 1, 4, /*space*/ 1,
                       IMAGE_SIZE * sizeof(float), TILE_WIDTH * sizeof(float),
                       TILE_HEIGHT, IMAGE_WIDTH * sizeof(float), 1, 0, 1, 0);

  air_queue_dispatch_and_wait(q, wr_idx, pkt_c);

  mlir_aie_print_dma_status(xaie, 7, 4);

  uint32_t errs = 0;
  for (int i = 0; i < IMAGE_SIZE; i++) {
    uint32_t d0;
    d0 = bank0_ptr[i];
    uint32_t d;
    d = bank1_ptr[i + IMAGE_SIZE];
    int r = i / IMAGE_WIDTH;
    int c = i % IMAGE_WIDTH;
    if ((r < TILE_HEIGHT) && (c < TILE_WIDTH)) {
      if (d0 != d) {
        printf("Part 0 %i [%d][%d]: Expect %08X, got %08X\n", i, r, c, d0, d);
        errs++;
      }
    } else if (d != 0xdeafcafe) {
      printf("Part X %i [%d][%d]: Expect %08X, got %08X\n", i, r, c, 0xdeafcafe,
             d);
      errs++;
    }
  }

  for (int i = 0; i < 64; i++) {
    uint32_t word0 = mlir_aie_read_buffer_buf0(xaie, i);
    uint32_t word1 = mlir_aie_read_buffer_buf1(xaie, i);

    printf("Tiles %x %08X %08X\r\n", i, word0, word1);
  }

  if (errs) {
    printf("FAIL: %d errors\n", errs);
    return -1;
  } else {
    printf("PASS!\n");
    return 0;
  }
}
