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

#include <xaiengine.h>

#include "air_host.h"
#include "air_tensor.h"

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

namespace air::partitions::partition_0 {
void mlir_aie_write_buffer_scratch_copy_0_0(aie_libxaie_ctx_t *, int, int32_t);
int32_t mlir_aie_read_buffer_scratch_copy_0_0(aie_libxaie_ctx_t *, int);
void mlir_aie_write_buffer_scratch_0_0(aie_libxaie_ctx_t *, int, int32_t);
int32_t mlir_aie_read_buffer_scratch_0_0(aie_libxaie_ctx_t *, int);
}; // namespace air::partitions::partition_0
using namespace air::partitions::partition_0;

int main(int argc, char *argv[]) {
  aie_libxaie_ctx_t *xaie = air_init_libxaie();

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  for (int i = 0; i < TILE_SIZE; i++) {
    mlir_aie_write_buffer_scratch_0_0(xaie, i, 0xfadefade);
    mlir_aie_write_buffer_scratch_copy_0_0(xaie, i, 0xabababab);
  }

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr =
      (uint32_t *)mmap(NULL, 0x20000, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                       AIR_VCK190_L2_DMA_BASE);

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

  auto graph_fn =
      (void (*)(void *, void *))dlsym((void *)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t, 2> input;
  tensor_t<uint32_t, 2> output;

  input.shape[0] = IMAGE_WIDTH;
  input.shape[1] = IMAGE_HEIGHT;
  output.shape[0] = IMAGE_WIDTH;
  output.shape[1] = IMAGE_HEIGHT;

  input.data = input.alloc = (uint32_t *)0;
  uint32_t *in = (uint32_t *)&bank0_ptr[0];

  output.data = output.alloc = (uint32_t *)(IMAGE_SIZE * sizeof(uint32_t));
  uint32_t *out = (uint32_t *)&bank0_ptr[IMAGE_SIZE];

  for (int i = 0; i < IMAGE_SIZE; i++) {
    in[i] = i + 0x1000;
    out[i] = 0x0defaced;
  }

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  int errors = 0;

  // Now look at the image, should have the bottom left filled in
  for (int i = 0; i < IMAGE_SIZE; i++) {
    u32 rb = out[i];

    u32 row = i / IMAGE_WIDTH;
    u32 col = i % IMAGE_WIDTH;

    if ((row >= TILE_HEIGHT) && (col < TILE_WIDTH)) {
      if (!(rb == 0x1000 + i)) {
        printf("IM %3d [%2d, %2d] should be %08X, is %08X. Tile %3d is %08X\n",
               i, col, row, i + 0x1000, rb, row * TILE_WIDTH + col - TILE_SIZE,
               mlir_aie_read_buffer_scratch_0_0(xaie, row * TILE_WIDTH + col -
                                                          TILE_SIZE));
        errors++;
      }
    } else {
      if (rb != 0x00defaced) {
        printf("IM %3d [%2d, %2d] should be 0xdefaced, is %08X\n", i, col, row,
               rb);
        errors++;
      }
    }
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", (TILE_SIZE + IMAGE_SIZE - errors),
           TILE_SIZE + IMAGE_SIZE);
    return -1;
  }

  free(input.alloc);
  free(output.alloc);
}
