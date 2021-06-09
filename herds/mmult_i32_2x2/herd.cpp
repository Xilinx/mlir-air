// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

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

#define VERBOSE 1
#define PROFILE 1

namespace {

air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie.inc"
#undef TileInst
#undef TileDMAInst

//
// global q ptr
//
queue_t *q = nullptr;
uint32_t *bram_ptr;

}

template<typename T>
void mm_out(tensor_t<T,2> *a, tensor_t<T,2> *b, tensor_t<T,2> *r)
{
  size_t a_h = a->shape[0];
  size_t a_w = a->shape[1];
  size_t b_h = b->shape[0];
  size_t b_w = b->shape[1];
  assert(a_w == b_h);

  for (size_t i=0; i<a_h; i++) {
    for (size_t j=0; j<b_w; j++) {
      size_t idx = i*b_w + j;
      r->d[idx] = (T)(0);
      for (size_t k=0, ke=a_w; k<a_w; k++) {
        T _a = a->d[i*a_w + k];
        T _b = b->d[k*b_w + j];
        r->d[idx] += _a * _b;
      }
    }
  }
}

int
main(int argc, char *argv[])
{
  uint64_t col = 7;
  uint64_t row = 2;

  xaie = air_init_libxaie1();

  if (VERBOSE)
    ACDC_print_tile_status(xaie->TileInst[col][2]);

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  if (VERBOSE)
    ACDC_print_tile_status(xaie->TileInst[col][2]);


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

  tensor_t<uint32_t,2> input_A;
  tensor_t<uint32_t,2> input_B;
  tensor_t<uint32_t,2> output;
  tensor_t<uint32_t,2> output_ref;

  input_A.shape[0] = input_A.shape[1] = 64;
  input_A.d = input_A.aligned = (uint32_t*)malloc(sizeof(uint32_t)*input_A.shape[0]*input_A.shape[1]);

  input_B.shape[0] = input_B.shape[1] = 64;
  input_B.d = input_B.aligned = (uint32_t*)malloc(sizeof(uint32_t)*input_B.shape[0]*input_B.shape[1]);

  output.shape[0] = output.shape[1] = 64;
  output.d = output.aligned = (uint32_t*)malloc(sizeof(uint32_t)*output.shape[0]*output.shape[1]);

  output_ref.shape[0] = output_ref.shape[1] = 64;
  output_ref.d = output_ref.aligned = (uint32_t*)malloc(sizeof(uint32_t)*output_ref.shape[0]*output_ref.shape[1]);
  
  printf("loading aie_ctrl.so\n");
  auto handle = air_herd_load_from_file("./aie_ctrl.so");
  assert(handle && "failed to open aie_ctrl.so");

  auto herd_fn = (void (*)(void*,void *,void*))dlsym((void*)handle, "_mlir_ciface_task");
  assert(herd_fn && "failed to locate _mlir_ciface_task in .so");

  for (int i=0; i<input_A.shape[0]*input_A.shape[1]; i++) {
    input_A.d[i] = (uint32_t)i;
    input_B.d[i] = (uint32_t)i+1.1f;
    output.d[i] = 0.0f;
    output_ref.d[i] = 0.0f;
  }

  mm_out(&input_A, &input_B, &output_ref);

  void *a, *b,*o;
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

  if (VERBOSE)
    ACDC_print_tile_status(xaie->TileInst[col][2]);

  for (int i=0; i<64; i++) {
    //printf("%d\n", mlir_read_buffer_buf0(i));
    //printf("%d\n", mlir_read_buffer_buf1(i));
    //printf("%d\n", mlir_read_buffer_buf2(i));
  }

  int errors = 0;
  auto output_size = output.shape[0]*output.shape[1];
  for (int i=0; i<output_size; i++) {
    float d = output.d[i];
    float ref = output_ref.d[i];
    if (d != ref) {
      errors++;
      printf("%04X: mismatch %f != %f\n", i, d, ref);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %ld/%ld.\n", (output_size-errors), output_size);
  }
}
