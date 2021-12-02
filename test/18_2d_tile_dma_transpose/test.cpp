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
#include <xaiengine.h>
#include "test_library.h"

#include "air_host.h"
#include "acdc_queue.h"
#include "hsa_defs.h"

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie_inc.cpp"
#undef TileInst
#undef TileDMAInst

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
  mlir_start_cores();

  uint32_t *bram_ptr;

  #define BRAM_ADDR 0x4000+AIR_VCK190_SHMEM_BASE
  #define DMA_COUNT 32

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
  }
 
  for (int i=0; i<DMA_COUNT; i++) {
    mlir_write_buffer_buf0(i, 0x12345670+i);
    bram_ptr[DMA_COUNT+i] = 0xdeface;
  }
  printf("Input Matrix:\n");
  int k = 0;
  for (int j=0; j < 4; j++) {
    for (int i=0; i < 8; i++) {
      bram_ptr[j*8 + i] = k++;
      printf("%2d\t",bram_ptr[j*8 + i]);
    }
    printf("\n");
  }
  printf("\n");

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  //
  // Set up a 1x3 herd starting 7,0
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 3);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *shim_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_device_init(shim_pkt,XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(q, wr_idx, shim_pkt);

  //
  // send the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, 0, col, 1, 0, 4, 2, BRAM_ADDR, DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);

  //
  // read the data
  //

  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt2 = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt2, 0, col, 0, 0, 4, 2, BRAM_ADDR+(DMA_COUNT*sizeof(float)), DMA_COUNT*sizeof(float), 1, 0, 1, 0, 1, 0);
  air_queue_dispatch_and_wait(q, wr_idx, pkt2);

  printf("Transpose Matrix in the Tile:\n");
  for (int j=0; j < 8; j++) {
    for (int i=0; i < 4; i++) {
      printf("%2d\t",mlir_read_buffer_buf0(j*4 + i));
    }
    printf("\n");
  }
  printf("\n");

  printf("Output Matrix:\n");
  for (int j=0; j < 4; j++) {
    for (int i=0; i < 8; i++) {
      printf("%2d\t",bram_ptr[j*8 + i + DMA_COUNT]);
    }
    printf("\n");
  }
  printf("\n");

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = bram_ptr[i + DMA_COUNT];
    ACDC_check("Check BRAM Result", d, i, errors);

    uint32_t d0 = mlir_read_buffer_buf0(i);
    uint32_t c = 0 + 
                 8 * ((i / 1) % 4) +
                 1 * ((i / 4) % 8);
    ACDC_check("Check Tile", d0, c, errors);
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, DMA_COUNT);
    return -1;
  }

}
