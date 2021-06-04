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

#include "air_host.h"
#include "air_tensor.h"
#include "test_library.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#define BRAM_ADDR 0x020100000000LL
#define DMA_COUNT 256

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie.0.inc"
#undef TileInst
#undef TileDMAInst

queue_t *q = nullptr;

}

extern "C" {

void _mlir_ciface_air_shim_memcpy(uint32_t id, uint64_t x, uint64_t y, void* t, uint64_t offset, uint64_t length) {
  uint64_t col = 2;

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, 0, col, 1, 0, 4, 2, BRAM_ADDR, DMA_COUNT*sizeof(uint32_t), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
}

}

int
main(int argc, char *argv[])
{
  uint64_t row = 0;
  uint64_t col = 7;

  xaie = air_init_libxaie1();

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();

  // create the queue
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  // Set up a 1x1 herd starting 7,2
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 1);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *shim_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(shim_pkt);
  shim_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  shim_pkt->arg[0]  = AIR_PKT_TYPE_DEVICE_INITIALIZE;
  shim_pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  shim_pkt->arg[0] |= ((uint64_t)XAIE_NUM_COLS << 40);
  air_queue_dispatch_and_wait(q, wr_idx, shim_pkt);

  printf("loading aie_ctrl.so\n");
  auto handle = dlopen("./aie_ctrl.so", RTLD_NOW);
  if (!handle) {
    printf("%s\n",dlerror());
  }
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void*))dlsym(handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> input;
  input.shape[0] = 256;
  input.d = (uint32_t*)malloc(sizeof(uint32_t)*256);
  for (int i=0; i<input.shape[0]; i++) {
    input.d[i] = i;
  }

  auto i = &input;
  graph_fn(i, i);

  ACDC_print_dma_status(xaie->TileInst[7][2]);

  // we copied the start of the shared bram into tile memory,
  // fish out the queue id and check it
  uint32_t d = mlir_read_buffer_buf0(24);
  printf("ID %x\n", d);

  if (d == 0xacdc)
    printf("PASS!\n");
  else
    printf("fail.\n");

}
