//===- herd.cpp -------------------------------------------------*- C++ -*-===//
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

#include <xaiengine.h>

#include "air_host.h"
#include "air_tensor.h"
#include "test_library.h"

#include <sys/time.h>

#define VERBOSE 0
#define PROFILE 0

namespace {

template<typename T>
void add_out(tensor_t<T,3> *a, tensor_t<T,3> *b, tensor_t<T,3> *r)
{
  size_t a_h = a->shape[0];
  size_t a_w = a->shape[1];
  size_t a_x = a->shape[2];
  size_t b_h = b->shape[0];
  size_t b_w = b->shape[1];
  assert(a_h == b_h);
  assert(a_w == b_w);

  for (size_t i=0; i<a_h*a_w*a_x; i++) {
    T _a = a->data[i];
    T _b = b->data[i];
    r->data[i] = _a + _b;
  }
}

}

namespace air::partitions::partition_0 {
int32_t mlir_aie_read_buffer_buf2(aie_libxaie_ctx_t *, int);
}
using namespace air::partitions::partition_0;

int
main(int argc, char *argv[])
{
  uint64_t row = 5;
  uint64_t col = 17;

  queue_t *q = nullptr;

  aie_libxaie_ctx_t *xaie = air_init_libxaie();

  if (VERBOSE)
    mlir_aie_print_tile_status(xaie,col,row);

  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 2, row, 2);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  dispatch_packet_t *dev_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_device_init(dev_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(q, wr_idx, dev_pkt);

  tensor_t<uint32_t,3> input_A;
  tensor_t<uint32_t,3> input_B;
  tensor_t<uint32_t,3> output;
  tensor_t<uint32_t,3> output_ref;

  #define M_SIZE 256

  input_A.shape[0] = 64;
  input_A.shape[1] = 64;
  input_A.shape[2] = 32;
  input_A.alloc = input_A.data = (uint32_t*)malloc(sizeof(uint32_t)*input_A.shape[0]*input_A.shape[1]*input_A.shape[2]);

  input_B.shape[0] = 64;
  input_B.shape[1] = 64;
  input_B.shape[2] = 32;
  input_B.alloc = input_B.data = (uint32_t*)malloc(sizeof(uint32_t)*input_B.shape[0]*input_B.shape[1]*input_B.shape[2]);

  output.shape[0] = 64;
  output.shape[1] = 64;
  output.shape[2] = 32;
  output.alloc = output.data = (uint32_t*)(size_t)malloc(sizeof(uint32_t)*output.shape[0]*output.shape[1]*output.shape[2]);

  output_ref.shape[0] = 64;
  output_ref.shape[1] = 64;
  output_ref.shape[2] = 32;
  output_ref.alloc = output_ref.data = (uint32_t*)malloc(sizeof(uint32_t)*output_ref.shape[0]*output_ref.shape[1]*output_ref.shape[2]);

  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open linked air module");

  auto herd_fn = (void (*)(void*,void *,void*))dlsym((void*)handle, "_mlir_ciface_forward");
  assert(herd_fn && "failed to locate _mlir_ciface_forward in .so");

  for (int i=0; i<input_A.shape[0]*input_A.shape[1]*input_A.shape[2]; i++) {
    input_A.data[i] = (rand() % 1024) + 1;
    input_B.data[i] = (rand() % 1024) + 1;
    output.data[i] = 0;
    output_ref.data[i] = 0;
  }

  add_out(&input_A, &input_B, &output_ref);

  void *a, *b, *o;
  a = &input_A;
  b = &input_B;
  o = &output;
  struct timeval before, after;
  long diff_s, diff_us;
  gettimeofday(&before, NULL);

  // run it
  herd_fn(a, b, o);

  gettimeofday(&after, NULL);
  diff_s = after.tv_sec - before.tv_sec;
  diff_us = after.tv_usec - before.tv_usec;

  if (diff_s)
    diff_us += 10000000;

  if (PROFILE) {
    printf("before %ld.%06ld\n",before.tv_sec, before.tv_usec);
    printf("after  %ld.%06ld\n",after.tv_sec, after.tv_usec);
    printf("diff   %ld.%06ld\n",diff_s, diff_us);
  }

  if (VERBOSE) {
    mlir_aie_print_tile_status(xaie,col,2);
    for (int i=0; i<64; i++) {
      printf("%d\n", mlir_aie_read_buffer_buf2(xaie, i));
    }
  }

  int errors = 0;
  auto output_size = output.shape[0]*output.shape[1];
  for (int i=0; i<output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref.data[i];
    if (d != ref) {
      errors++;
      if (errors < 100)
      printf("%04X: mismatch %d != %d\n", i, d, ref);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %ld/%ld.\n", (output_size-errors), output_size);
  }
}
