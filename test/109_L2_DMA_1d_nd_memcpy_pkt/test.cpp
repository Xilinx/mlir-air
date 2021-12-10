// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cstdio>
#include <climits>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "air_host.h"

#include "aie_inc.cpp"

#define XFR_SZ 64

int main(int argc, char *argv[])
{
  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);
  
  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  for (int i=0; i<XFR_SZ; i++) {
    mlir_aie_write_buffer_buf1(xaie, i,i+0xacdc1000);
    mlir_aie_write_buffer_buf2(xaie, i,i+0xacdc2000);
  }
  for (int i=0; i<XFR_SZ; i++) {
    uint32_t word0 = mlir_aie_read_buffer_buf1(xaie, i);
    uint32_t word1 = mlir_aie_read_buffer_buf2(xaie, i);

    printf("Tiles %x %08X %08X\r\n", i, word0, word1);
  }

  mlir_aie_print_dma_status(xaie, 7, 3);
  mlir_aie_print_dma_status(xaie, 7, 4);
 
 int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE);
  uint32_t *bank1_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+0x20000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 2 for the upper memory as it goes in
  for (int i=0;i<2*XFR_SZ;i++) {
    // 3D DMA address generation
    //           X Y Z
    // increment 1 2 8
    // wrap      2 4 max
    // offset    4 1 8
    int an = 4*((i/1)%2) + 1*((i/2)%4) + 8*((i/8)%UINT_MAX); 
    uint32_t toWrite = i;

    bank1_ptr[an] = toWrite + (2 << 28);
    bank0_ptr[an] = toWrite + (1 << 28);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

  //
  // Set up a 1x2 herd starting 7,3
  //
  air_packet_herd_init(pkt, 0, 7, 1, 3, 2);

  // dispatch packet
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
    
  // globally bypass headers
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

  l2_dma_cmd_t cmd;
  cmd.select = 7;
  cmd.length = 0;
  cmd.uram_addr = 1;
  cmd.id = 0;

  uint64_t stream = 0;
  air_packet_l2_dma(pkt, stream, cmd);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  for (int sel=0; sel<2; sel++) {
    //
    // send the data
    //
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  
    air_packet_nd_memcpy(pkt, 0, 7, 1, sel, 4, 1, 
                         0,
                         XFR_SZ*sizeof(float), 
                         1, 0,
                         1, 0,
                         1, 0);

    //
    // read the data back
    //
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

    air_packet_nd_memcpy(pkt, 0, 7, 0, sel, 4, 1, 
                         XFR_SZ*sizeof(float), 
                         XFR_SZ*sizeof(float), 
                         1, 0,
                         1, 0,
                         1, 0);
  }

  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  sleep(1);
  mlir_aie_print_dma_status(xaie, 7, 3);
  mlir_aie_print_dma_status(xaie, 7, 4);
  
  uint32_t errs = 0;
  for (int i=XFR_SZ; i<2*XFR_SZ; i++) {
    uint32_t d0;
    d0 = bank0_ptr[i-XFR_SZ];
    uint32_t d;
    d = bank0_ptr[i];
    if (d != d0) {
      printf("Part 0 %i : Expect %08X, got %08X\n",i, d0, d);
      errs++;
    }
  }
  for (int i=XFR_SZ; i<2*XFR_SZ; i++) {
    uint32_t d0;
    d0 = bank1_ptr[i-XFR_SZ];
    uint32_t d;
    d = bank1_ptr[i];
    if (d != d0) {
      printf("Part 1 %i : Expect %08X, got %08X\n",i, d0, d);
      errs++;
    }
  }

  for (int i=0; i<16; i++) {
    uint32_t word0 = mlir_aie_read_buffer_buf1(xaie, i);
    uint32_t word1 = mlir_aie_read_buffer_buf2(xaie, i);

    printf("Tiles %x %08X %08X\r\n", i, word0, word1);
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
