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

#include "aie_inc.cpp"

#define DMA_COUNT 32

int main(int argc, char *argv[]) {
  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();

  for (int i = 0; i < DMA_COUNT; i++) {
    mlir_aie_write_buffer_buf0(xaie, i, 0xcfcfcfcf);
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

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it
  // goes in
  for (int i = 0; i < 2 * DMA_COUNT; i++) {
    uint32_t toWrite = i + 0xacdc00;

    bank1_ptr[i] = toWrite + (1 << 28);
    bank0_ptr[i] = toWrite + (2 << 28);
  }

  // Read back the values from above
  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t word = mlir_aie_read_buffer_buf0(xaie, i);
    uint32_t word0 = bank0_ptr[i];
    uint32_t word1 = bank1_ptr[i];

    printf("[%2d] Tile: %08X\tL2: %08X %08X\r\n", i, word, word0, word1);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

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

  auto graph_fn = (void (*)(void *, void *, void *, void *))dlsym(
      (void *)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t, 1> input;
  tensor_t<uint32_t, 1> output;
  input.shape[0] = DMA_COUNT;
  output.shape[0] = DMA_COUNT;

  input.data = input.alloc =
      (uint32_t *)malloc(sizeof(uint32_t) * input.shape[0]);
  uint32_t *in = (uint32_t *)input.data;
  output.data = output.alloc =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0]);
  uint32_t *out = (uint32_t *)output.data;

  for (int i = 0; i < input.shape[0]; i++) {
    in[i] = i;
    out[i] = 0xcafe;
  }

  tensor_t<uint32_t, 1> l2_input;
  tensor_t<uint32_t, 1> l2_output;
  l2_input.shape[0] = DMA_COUNT;
  l2_output.shape[0] = DMA_COUNT;
  l2_input.data = l2_input.alloc = (uint32_t *)(0);
  l2_output.data = l2_output.alloc =
      (uint32_t *)(sizeof(uint32_t) * input.shape[0]);

  auto i = &input;
  auto o = &output;
  auto i2 = &l2_input;
  auto o2 = &l2_output;
  printf("Running air test...\n");
  graph_fn(i, o, i2, o2);
  printf("\t...done.\n");

  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t word = mlir_aie_read_buffer_buf0(xaie, i);
    uint32_t word0 = bank0_ptr[i];
    uint32_t word1 = bank1_ptr[i];

    printf("[%2d] Tile: %08X\tL2: %08X %08X\r\n", i, word, word0, word1);
  }
  for (int i = DMA_COUNT; i < 2 * DMA_COUNT; i++) {
    uint32_t word0 = bank0_ptr[i];
    uint32_t word1 = bank1_ptr[i];

    printf("[%2d] \t\t\tL2: %08X %08X\r\n", i, word0, word1);
  }

  int errors = 0;
  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t d = out[i];
    printf("[%2d] DDR: %08X\r\n", i, d);
    if (d != i) {
      errors++;
      printf("mismatch %x != %x\n", d, i);
    }
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", (DMA_COUNT - errors), DMA_COUNT);
    return -1;
  }
}
