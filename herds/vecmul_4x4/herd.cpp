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
#define GRID_SIZE 16384

namespace air::herds::herd_0 {

int32_t mlir_aie_read_buffer_buf0(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf1(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf2(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf3(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf4(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf5(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf6(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf7(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf8(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf9(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf10(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_buf11(aie_libxaie_ctx_t*, int);
void mlir_aie_write_buffer_buf0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf3(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf4(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf5(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf6(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf7(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf8(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf9(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf10(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_buf11(aie_libxaie_ctx_t*, int, int32_t);
}
using namespace air::herds::herd_0;

int
main(int argc, char *argv[])
{
  uint64_t col = 20;
  uint64_t row = 3;

  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  if (VERBOSE)
    mlir_aie_print_tile_status(xaie, col, row);

  // Stomp
  for (int i=0; i<64; i++) {
    mlir_aie_write_buffer_buf0(xaie, i, 0x0decaf);
    mlir_aie_write_buffer_buf1(xaie, i, 0x1decaf);
    mlir_aie_write_buffer_buf2(xaie, i, 0x2decaf);
    mlir_aie_write_buffer_buf3(xaie, i, 0x3decaf);
    mlir_aie_write_buffer_buf4(xaie, i, 0x4decaf);
    mlir_aie_write_buffer_buf5(xaie, i, 0x5decaf);
    mlir_aie_write_buffer_buf6(xaie, i, 0x6decaf);
    mlir_aie_write_buffer_buf7(xaie, i, 0x7decaf);
    mlir_aie_write_buffer_buf8(xaie, i, 0x8decaf);
    mlir_aie_write_buffer_buf9(xaie, i, 0x9decaf);
    mlir_aie_write_buffer_buf10(xaie, i, 0xadecaf);
    mlir_aie_write_buffer_buf11(xaie, i, 0xbdecaf);
  }

  if (VERBOSE)
    mlir_aie_print_tile_status(xaie, col, row);

  // create the queue
  
  queue_t *q = nullptr;

  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // uint64_t wr_idx = queue_add_write_index(q, 1);
  // uint64_t packet_id = wr_idx % q->size;

  // dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  // air_packet_herd_init(herd_pkt, 0, col, 4, row, 4);
  // air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // wr_idx = queue_add_write_index(q, 1);
  // packet_id = wr_idx % q->size;

  // dispatch_packet_t *dev_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  // air_packet_device_init(dev_pkt, XAIE_NUM_COLS);
  // air_queue_dispatch_and_wait(q, wr_idx, dev_pkt);

  tensor_t<int32_t,1> input_A;
  tensor_t<int32_t,1> input_B;
  tensor_t<int32_t,1> output;

  input_A.shape[0] = GRID_SIZE;
  input_A.d = input_A.aligned = (int32_t*)malloc(sizeof(int32_t)*input_A.shape[0]);

  input_B.shape[0] = GRID_SIZE;
  input_B.d = input_B.aligned = (int32_t*)malloc(sizeof(int32_t)*input_B.shape[0]);

  output.shape[0] = GRID_SIZE;
  output.d = output.aligned = (int32_t*)malloc(sizeof(int32_t)*output.shape[0]);
  
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open linked air module");

  auto herd_fn = (void (*)(void*,void *,void*))dlsym((void*)handle, "_mlir_ciface_forward");
  assert(herd_fn && "failed to locate _mlir_ciface_forward in vecmul.so");

  for (int i=0; i<input_A.shape[0]; i++) {
    input_A.d[i] = i;
    input_B.d[i] = i+1;
    output.d[i] = 0xfeedcafe;
  }

  if (VERBOSE) {
    for (int i=0; i<16; i++) { 
      int32_t rb0 = mlir_aie_read_buffer_buf0(xaie, i);
      int32_t rb1 = mlir_aie_read_buffer_buf1(xaie, i);
      int32_t rb2 = mlir_aie_read_buffer_buf2(xaie, i);
      int32_t rb3 = mlir_aie_read_buffer_buf3(xaie, i);
      int32_t rb4 = mlir_aie_read_buffer_buf4(xaie, i);
      int32_t rb5 = mlir_aie_read_buffer_buf5(xaie, i);
      int32_t rb6 = mlir_aie_read_buffer_buf6(xaie, i);
      int32_t rb7 = mlir_aie_read_buffer_buf7(xaie, i);
      int32_t rb8 = mlir_aie_read_buffer_buf8(xaie, i);
      int32_t rb9 = mlir_aie_read_buffer_buf9(xaie, i);
      int32_t rb10 = mlir_aie_read_buffer_buf10(xaie, i);
      int32_t rb11 = mlir_aie_read_buffer_buf11(xaie, i);
      printf("before %d [7][2] : %08X * %08X -> %08X, [8][2] :%08X * %08X -> %08X, [7][3] : %08X * %08X -> %08X, [8][3] :%08X * %08X-> %08X\n", i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8, rb9, rb10, rb11);
    }
  }

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
  if (VERBOSE) {
    for (int i=0; i<16; i++) { 
      int32_t rb0 = mlir_aie_read_buffer_buf0(xaie, i);
      int32_t rb1 = mlir_aie_read_buffer_buf1(xaie, i);
      int32_t rb2 = mlir_aie_read_buffer_buf2(xaie, i);
      int32_t rb3 = mlir_aie_read_buffer_buf3(xaie, i);
      int32_t rb4 = mlir_aie_read_buffer_buf4(xaie, i);
      int32_t rb5 = mlir_aie_read_buffer_buf5(xaie, i);
      int32_t rb6 = mlir_aie_read_buffer_buf6(xaie, i);
      int32_t rb7 = mlir_aie_read_buffer_buf7(xaie, i);
      int32_t rb8 = mlir_aie_read_buffer_buf8(xaie, i);
      int32_t rb9 = mlir_aie_read_buffer_buf9(xaie, i);
      int32_t rb10 = mlir_aie_read_buffer_buf10(xaie, i);
      int32_t rb11 = mlir_aie_read_buffer_buf11(xaie, i);
      printf(" after %d [7][2] : %08X * %08X -> %08X, [8][2] :%08X * %08X -> %08X, [7][3] : %08X * %08X -> %08X, [8][3] :%08X * %08X-> %08X\n", i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8, rb9, rb10, rb11);
    }
    mlir_aie_print_tile_status(xaie, col, row);
  }

  int errors = 0;
  for (int i=0; i<output.shape[0]; i++) {
    uint32_t a = input_A.d[i];
    uint32_t b = input_B.d[i];
    uint32_t d = output.d[i];
    if (d != (a*b)) {
      errors++;
      if (errors < 10)
        printf("%04X: mismatch %x != %x * %x (%x)\n", i, d, a, b, (a*b));
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %ld/%ld.\n", (output.shape[0]-errors), output.shape[0]);
  }
}
