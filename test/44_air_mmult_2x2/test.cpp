// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

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
#include "test_library.h"

#include <sys/time.h>

#define VERBOSE 1
#define PROFILE 0

namespace {

template <typename T>
void mm_out(tensor_t<T, 2> *a, tensor_t<T, 2> *b, tensor_t<T, 2> *r) {
  size_t a_h = a->shape[0];
  size_t a_w = a->shape[1];
  size_t b_h = b->shape[0];
  size_t b_w = b->shape[1];
  assert(a_w == b_h);

  for (size_t i = 0; i < a_h; i++) {
    for (size_t j = 0; j < b_w; j++) {
      size_t idx = i * b_w + j;
      r->data[idx] = (T)(0);
      for (size_t k = 0, ke = a_w; k < a_w; k++) {
        T _a = a->data[i * a_w + k];
        T _b = b->data[k * b_w + j];
        r->data[idx] += _a * _b;
      }
    }
  }
}

} // namespace

namespace air::partitions::partition0 {
int32_t mlir_aie_read_buffer_buf0(aie_libxaie_ctx_t *, int);
int32_t mlir_aie_read_buffer_buf11(aie_libxaie_ctx_t *, int);
int32_t mlir_aie_read_buffer_buf2(aie_libxaie_ctx_t *, int);
} // namespace air::partitions::partition0
using namespace air::partitions::partition0;

int main(int argc, char *argv[]) {
  uint64_t col = 5;
  uint64_t row = 3;

  queue_t *q = nullptr;

  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  if (VERBOSE)
    mlir_aie_print_tile_status(xaie, col, row);

  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // uint64_t wr_idx = queue_add_write_index(q, 1);
  // uint64_t packet_id = wr_idx % q->size;

  // dispatch_packet_t *herd_pkt =
  //     (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  // air_packet_herd_init(herd_pkt, 0, col, 2, row, 2);
  // air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // wr_idx = queue_add_write_index(q, 1);
  // packet_id = wr_idx % q->size;

  // dispatch_packet_t *dev_pkt =
  //     (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
  // air_packet_device_init(dev_pkt, XAIE_NUM_COLS);
  // air_queue_dispatch_and_wait(q, wr_idx, dev_pkt);

  tensor_t<uint32_t, 2> input_A;
  tensor_t<uint32_t, 2> input_B;
  tensor_t<uint32_t, 2> output;
  tensor_t<uint32_t, 2> output_ref0;

#define M_SIZE 64

  input_A.shape[0] = input_A.shape[1] = M_SIZE;
  input_A.alloc = input_A.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_A.shape[0] * input_A.shape[1]);

  input_B.shape[0] = input_B.shape[1] = M_SIZE;
  input_B.alloc = input_B.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_B.shape[0] * input_B.shape[1]);

  output.shape[0] = output.shape[1] = M_SIZE;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0] * output.shape[1]);

  output_ref0.shape[0] = output_ref0.shape[1] = M_SIZE;
  output_ref0.alloc = output_ref0.data = (uint32_t *)malloc(
      sizeof(uint32_t) * output_ref0.shape[0] * output_ref0.shape[1]);

  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open linked air module");

  auto herd_fn = (void (*)(void *, void *, void *))dlsym(
      (void *)handle, "_mlir_ciface_forward");
  assert(herd_fn && "failed to locate _mlir_ciface_forward in .so");

  for (int i = 0; i < input_A.shape[0] * input_A.shape[1]; i++) {
    input_A.data[i] = (rand() % 1024) + 1;
    input_B.data[i] = (rand() % 1024) + 1;
    output.data[i] = 0;
    output_ref0.data[i] = 0;
  }

  mm_out(&input_A, &input_B, &output_ref0);

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
    printf("before %ld.%06ld\n", before.tv_sec, before.tv_usec);
    printf("after  %ld.%06ld\n", after.tv_sec, after.tv_usec);
    printf("diff   %ld.%06ld\n", diff_s, diff_us);
  }

  if (VERBOSE) {
    //    mlir_aie_print_tile_status(xaie,col,2);
    for (int i = 0; i < 32; i++) {
      //      printf("%d ", mlir_aie_read_buffer_buf0(xaie, i));
      //      printf("%d\n", mlir_aie_read_buffer_buf11(xaie, i));
      printf("%d\n", mlir_aie_read_buffer_buf2(xaie, i));
    }
  }

  int errors = 0;
  auto output_size = output.shape[0] * output.shape[1];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref0.data[i];
    if (d != ref) {
      errors++;
      if (errors < 100)
        printf("%04X: mismatch %d != %d\n", i, d, ref);
    }
  }

  free(input_A.alloc);
  free(input_B.alloc);
  free(output.alloc);
  free(output_ref0.alloc);

  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
    return -1;
  }
  return 0;
}
