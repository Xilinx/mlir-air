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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dlfcn.h>

#include "air_host.h"
#include "air_tensor.h"

#define DMA_COUNT 256

namespace air::partitions::partition_0 {
void mlir_aie_write_buffer_buf0(aie_libxaie_ctx_t*, int, int32_t);
};
using namespace air::partitions::partition_0;

int
main(int argc, char *argv[])
{
  uint64_t row = 2;
  uint64_t col = 7;

  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  for (int i=0; i<DMA_COUNT; i++)
    mlir_aie_write_buffer_buf0(xaie, i, i+0x10);

  uint32_t *bram_ptr = nullptr;

  // use BRAM_ADDR + 0x4000 as the data address
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_BBUFF_BASE);
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = 0xdeadbeef;    }
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> input;
  input.shape[0] = DMA_COUNT;
  input.alloc = input.data = (uint32_t *)malloc(sizeof(uint32_t) * DMA_COUNT);
  for (int i=0; i<input.shape[0]; i++) {
    input.data[i] = i + 0x10;
  }

  mlir_aie_print_dma_status(xaie, col, row);

  auto i = &input;
  graph_fn(i);

  mlir_aie_print_dma_status(xaie, col, row);
  mlir_aie_print_tile_status(xaie, col, row);

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[i];
    if (d != (i+0x10)) {
      errors++;
      printf("mismatch %x != 0x10 + %x\n", d, i);
    }
  }

  free(input.alloc);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT-errors), DMA_COUNT);
    return -1;
  }

}
