// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

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

namespace air::herds::herd_0 {
int32_t mlir_aie_read_buffer_scratch_0_0(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_scratch_copy_0_0(aie_libxaie_ctx_t*, int);
void mlir_aie_write_buffer_scratch_0_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_copy_0_0(aie_libxaie_ctx_t*, int, int32_t);
};
using namespace air::herds::herd_0;

#define DMA_COUNT 16

int
main(int argc, char *argv[])
{
  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  for (int i=0; i<DMA_COUNT; i++) {
    mlir_aie_write_buffer_scratch_0_0(xaie, i, 0xfadefade);
    mlir_aie_write_buffer_scratch_copy_0_0(xaie, i, 0xfadefade);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");
  
  //
  // Set up a 1x1 herd starting 7,2
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, 7, 1, 2, 1);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  printf("loading air module\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open air module");

  auto graph_fn = (void (*)(void*,void *))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t,1> input;
  tensor_t<uint32_t,1> output;

  input.shape[0] = DMA_COUNT;
  output.shape[0] = DMA_COUNT;

  XAieLib_MemInst *mem_i = XAieLib_MemAllocate(sizeof(uint32_t)*input.shape[0], 0);
  input.d = input.aligned = (uint32_t*)XAieLib_MemGetPaddr(mem_i);
  uint32_t *in = (uint32_t*)XAieLib_MemGetVaddr(mem_i); 

  XAieLib_MemInst *mem_o = XAieLib_MemAllocate(sizeof(uint32_t)*output.shape[0], 0);
  output.d = output.aligned = (uint32_t*)XAieLib_MemGetPaddr(mem_o);
  uint32_t *out = (uint32_t*)XAieLib_MemGetVaddr(mem_o);

  if (mem_i) {
    for (int i=0; i<DMA_COUNT; i++) {
      in[i] = i+0x1;
      //printf("in %p %p %llx\n", &in[i], &input.d[i], in[i]);
    }
  } else {
    printf("ERROR: could not allocate memory!\n");
    return 1;
  }
  if (mem_o) {
    for (int i=0; i<DMA_COUNT; i++) {
      out[i] = 0x00defaced;
      //printf("out %p %p %llx\n", &out[i], &output.d[i], out[i]);
    }
  } else {
    printf("ERROR: could not allocate memory!\n");
    return 1;
  }

  XAieLib_MemSyncForDev(mem_i);
  XAieLib_MemSyncForDev(mem_o);

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d0 = mlir_aie_read_buffer_scratch_0_0(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_scratch_copy_0_0(xaie, i);
    if (d0+1 != d1) {
      printf("mismatch tile %x != %x\n", d0, d1);
      errors++;
    }
  }

  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = out[i];
    if (d != (i+2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  XAieLib_MemFree(mem_i);
  XAieLib_MemFree(mem_o);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, 2*DMA_COUNT);
    return -1;
  }
}
