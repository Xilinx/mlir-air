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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include "air_host.h"
#include "test_library.h"

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 64
#define TILE_HEIGHT 64
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

#define SUB_TILE_WIDTH 32
#define SUB_TILE_HEIGHT 32
#define SUB_TILE_SIZE (SUB_TILE_WIDTH * SUB_TILE_HEIGHT)

namespace air::partitions::partition_0 {
void mlir_aie_write_buffer_scratch_a_0_0(aie_libxaie_ctx_t *, int, int32_t);
int32_t mlir_aie_read_buffer_scratch_a_0_0(aie_libxaie_ctx_t *, int);
void mlir_aie_write_buffer_scratch_b_0_0(aie_libxaie_ctx_t *, int, int32_t);
int32_t mlir_aie_read_buffer_scratch_b_0_0(aie_libxaie_ctx_t *, int);
void mlir_aie_write_buffer_scratch_c_0_0(aie_libxaie_ctx_t *, int, int32_t);
int32_t mlir_aie_read_buffer_scratch_c_0_0(aie_libxaie_ctx_t *, int);
}; // namespace air::partitions::partition_0
using namespace air::partitions::partition_0;

int main(int argc, char *argv[]) {
  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  for (int i = 0; i < SUB_TILE_SIZE; i++) {
    mlir_aie_write_buffer_scratch_a_0_0(xaie, i, 0xA0000 + i);
    mlir_aie_write_buffer_scratch_b_0_0(xaie, i, 0xB0000 + i);
    mlir_aie_write_buffer_scratch_c_0_0(xaie, i, 0xC0000 + i);
  }

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE);
  uint32_t *bank1_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE + 0x20000);

  //
  // Set up a 1x1 herd starting 7,4
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt =
      (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 1, 4, 1);
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

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void *, void *, void *, void *, void *,
                            void *))dlsym((void *)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t, 2> input_a;
  tensor_t<uint32_t, 2> input_b;
  tensor_t<uint32_t, 2> inout_c;
  tensor_t<uint32_t, 2> l2_a;
  tensor_t<uint32_t, 2> l2_b;
  tensor_t<uint32_t, 2> l2_c;

  input_a.shape[0] = IMAGE_WIDTH;
  input_a.shape[1] = IMAGE_HEIGHT;
  input_b.shape[0] = IMAGE_WIDTH;
  input_b.shape[1] = IMAGE_HEIGHT;
  inout_c.shape[0] = IMAGE_WIDTH;
  inout_c.shape[1] = IMAGE_HEIGHT;
  l2_a.shape[0] = TILE_WIDTH;
  l2_a.shape[1] = TILE_HEIGHT;
  l2_b.shape[0] = TILE_WIDTH;
  l2_b.shape[1] = TILE_HEIGHT;
  l2_c.shape[0] = TILE_WIDTH;
  l2_c.shape[1] = TILE_HEIGHT;

  input_a.alloc = input_a.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_a.shape[0] * input_a.shape[1]);
  uint32_t *a = (uint32_t *)input_a.data;
  input_b.alloc = input_b.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_b.shape[0] * input_b.shape[1]);
  uint32_t *b = (uint32_t *)input_b.data;
  inout_c.alloc = inout_c.data = (uint32_t *)malloc(
      sizeof(uint32_t) * inout_c.shape[0] * inout_c.shape[1]);
  uint32_t *c = (uint32_t *)inout_c.data;

  for (int i = 0; i < IMAGE_SIZE; i++) {
    a[i] = i + 1;
    b[i] = i + 2;
    c[i] = 0;
  }

  l2_a.alloc = l2_a.data = (uint32_t *)0;
  uint32_t *l2a = (uint32_t *)&bank0_ptr[0];
  l2_b.alloc = l2_b.data = (uint32_t *)0x20000;
  uint32_t *l2b = (uint32_t *)&bank1_ptr[0];
  l2_c.alloc = l2_c.data = (uint32_t *)(TILE_SIZE * sizeof(uint32_t));
  uint32_t *l2c = (uint32_t *)&bank0_ptr[TILE_SIZE];

  for (int i = 0; i < TILE_SIZE; i++) {
    l2a[i] = 0xA20000;
    l2b[i] = 0xB20000;
    l2c[i] = 0xC20000;
  }

  void *ia, *ib, *ic, *a2, *b2, *c2;
  ia = &input_a;
  ib = &input_b;
  ic = &inout_c;
  a2 = &l2_a;
  b2 = &l2_b;
  c2 = &l2_c;
  graph_fn(ia, ib, ic, a2, b2, c2);

  int errors = 0;

  // Now look at the image, should have the bottom left filled in
  for (int i = 0; i < IMAGE_SIZE; i++) {
    u32 d = c[i];

    if (d != ((i + 1) + (i + 2))) {
      errors++;
      printf("mismatch %x != %x + %x\n", d, i + 1, i + 2);
    }
  }

  // for (int i=0; i<TILE_SIZE; i++) {
  //   printf("L2 A: %X B: %X C: %X\n",l2a[i],l2b[i],l2c[i]);
  // }
  // for (int i=0; i<SUB_TILE_SIZE; i++) {
  //   int32_t sa = mlir_aie_read_buffer_scratch_a_0_0(xaie,i);
  //   int32_t sb = mlir_aie_read_buffer_scratch_b_0_0(xaie,i);
  //   int32_t sc = mlir_aie_read_buffer_scratch_c_0_0(xaie,i);
  //   printf("Tile A: %X B: %X C: %X\n",sa,sb,sc);
  // }

  free(input_a.alloc);
  free(input_b.alloc);
  free(inout_c.alloc);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", (IMAGE_SIZE - errors), IMAGE_SIZE);
    return -1;
  }
}
