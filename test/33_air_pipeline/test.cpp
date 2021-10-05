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
#include <xaiengine.h>
#include "test_library.h"

#include "air_host.h"
#include "air_tensor.h"

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "acdc_project/aie_inc.cpp"
#undef TileInst
#undef TileDMAInst

}

int
main(int argc, char *argv[])
{
  auto col = 3;
  auto row = 3;

  xaie = air_init_libxaie1();

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  uint32_t *bram_ptr = nullptr;

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1)
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE+0x4000);

  if (!bram_ptr)
    return -1;

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  // herd_setup packet
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 4, row, 1);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // device init packet
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *shim_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt,XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(q, wr_idx, shim_pkt);

  #define DATA_LENGTH 1024
  #define DATA_TYPE int

  tensor_t<DATA_TYPE,1> input_a, input_b;
  tensor_t<DATA_TYPE,1> output;
  input_a.shape[0] = DATA_LENGTH;
  input_a.d = input_a.aligned = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*input_a.shape[0]);

  input_b.shape[0] = DATA_LENGTH;
  input_b.d = input_b.aligned = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*input_b.shape[0]);

  output.shape[0] = input_a.shape[0];
  output.d = output.aligned = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*output.shape[0]);
  DATA_TYPE d = 1;
  for (int i=0; i<input_a.shape[0]; i++) {
    input_a.d[i] = d;
    input_b.d[i] = ((DATA_TYPE)DATA_LENGTH)+d;
    output.d[i] = -1;
    d += 1;
  }

  for (int i=0;i<DATA_LENGTH;i++) {
    bram_ptr[i] = input_a.d[i];
    bram_ptr[i+DATA_LENGTH] = input_b.d[i];
    bram_ptr[i+2*DATA_LENGTH] = -42;
  }

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_a = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_a, 0, 2, 1, 0, 4, 2, AIR_VCK190_SHMEM_BASE+0x4000, DATA_LENGTH*sizeof(DATA_TYPE), 1, 0, 1, 0, 1, 0);

  air_queue_dispatch_and_wait(q, wr_idx, pkt_a);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_b = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_b, 0, 2, 1, 1, 4, 2, AIR_VCK190_SHMEM_BASE+0x4000+DATA_LENGTH*sizeof(DATA_TYPE), DATA_LENGTH*sizeof(DATA_TYPE), 1, 0, 1, 0, 1, 0);

  air_queue_dispatch_and_wait(q, wr_idx, pkt_b);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt_c = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt_c, 0, 2, 0, 0, 4, 2, AIR_VCK190_SHMEM_BASE+0x4000+2*DATA_LENGTH*sizeof(DATA_TYPE), DATA_LENGTH*sizeof(DATA_TYPE), 1, 0, 1, 0, 1, 0);

  air_queue_dispatch_and_wait(q, wr_idx, pkt_c);

  int errors = 0;

  for (int i=0;i<DATA_LENGTH;i++) {
    output.d[i] = bram_ptr[i+2*DATA_LENGTH];
    DATA_TYPE ref = (input_a.d[i]*input_b.d[i]) + (DATA_TYPE)1 + (DATA_TYPE)2 + (DATA_TYPE)3;
    if (output.d[i] != ref) {
      printf("output[%d] = %d (expected %d)\n", i, output.d[i], ref);
      errors++;
    }
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (DATA_LENGTH-errors), DATA_LENGTH);
    return -1;
  }
}
