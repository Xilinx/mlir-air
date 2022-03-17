// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cstdio>
#include <climits>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dlfcn.h>

#include "air_host.h"
#include "air_tensor.h"

#define DMA_COUNT 32

namespace air::herds::herd_0 {
void mlir_aie_write_buffer_buf0(aie_libxaie_ctx_t*, int, int32_t);
int32_t mlir_aie_read_buffer_buf0(aie_libxaie_ctx_t*, int);
};
using namespace air::herds::herd_0;

int main(int argc, char *argv[])
{
  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  for (int i=0; i<16; i++) {
    mlir_aie_write_buffer_buf0(xaie, i, 0xdecaf);
    mlir_aie_write_buffer_buf0(xaie, i+16, 0xacafe);
  }

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE);
  uint32_t *bank1_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+0x20000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it goes in
  for (int i=0;i<32;i++) {
    uint32_t toWrite = i + 0x0112;

    bank0_ptr[i] = toWrite + (1 << 28);
    bank1_ptr[i] = toWrite + (2 << 28);
  }

  // Read back the values from above
  for (int i=0;i<32;i++) {
    uint32_t word0 = bank0_ptr[i];
    uint32_t word1 = bank1_ptr[i];

    printf("%d: %08X %08X\r\n", i, word0, word1);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  //
  // Set up a 4x4 herd starting 7,1
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 4, 1, 4);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
  
  // globally bypass headers
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

  static l2_dma_cmd_t cmd;
  cmd.select = 7;
  cmd.length = 0;
  cmd.uram_addr = 1;
  cmd.id = 0;

  uint64_t stream = 0;
  air_packet_l2_dma(pkt, stream, cmd);

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> input;
  input.shape[0] = DMA_COUNT;
  input.d = input.aligned = (uint32_t*)(0);

  auto i = &input;
  graph_fn(i);

  sleep(1);
  
  uint32_t errs = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d;
    d = mlir_aie_read_buffer_buf0(xaie, i);
    printf("%d: %08X\n", i, d);
    if ((d & 0x0fffffff) != (i+0x0112)) {
      printf("Word %i : Expect %08X, got %08X\n",i, i+0x0112, d);
      errs++;
    }
  }

  if (errs) {
    printf("FAIL: %d errors\n", errs);
    return -1;
  }
  else {
    printf("PASS!\n");
    return 0;
  }
}
