// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "air_host.h"

#include "aie_inc.cpp"

int main(int argc, char *argv[])
{
  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  for (int i=0; i<32; i++) {
    mlir_aie_write_buffer_a(xaie, i, 0xcafe01);
    mlir_aie_write_buffer_b(xaie, i, 0xcafe02);
    mlir_aie_write_buffer_c(xaie, i, 0xcafe03);
    mlir_aie_write_buffer_d(xaie, i, 0xcafe04);
    mlir_aie_write_buffer_e(xaie, i, 0xcafe05);
    mlir_aie_write_buffer_f(xaie, i, 0xcafe06);
    mlir_aie_write_buffer_g(xaie, i, 0xcafe07);
    mlir_aie_write_buffer_i(xaie, i, 0xcafe08);
  }

  mlir_aie_print_dma_status(xaie, 7, 2);
  mlir_aie_print_dma_status(xaie, 8, 2);
  mlir_aie_print_dma_status(xaie, 9, 2);
  mlir_aie_print_dma_status(xaie, 10, 2);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return -1;

  uint32_t *bank0_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+0*0x20000);
  uint32_t *bank1_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+1*0x20000);
  uint32_t *bank2_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+2*0x20000);
  uint32_t *bank3_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+3*0x20000);
  uint32_t *bank4_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+4*0x20000);
  uint32_t *bank5_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+5*0x20000);
  uint32_t *bank6_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+6*0x20000);
  uint32_t *bank7_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+7*0x20000);

  // Write an ascending pattern value into the memories
  // Also stamp with 1 for the lower memory, and 1 for the upper memory as it goes in
  for (int i=0;i<32;i++) {
    uint32_t upper_lower = (i%8)/4;
    uint32_t first128_second128 = i%2;
    uint32_t first64_second64 = (i%16)/8;
    uint32_t first32_second32 = (i/2)%2;
    uint32_t offset = (first128_second128)*4;
    offset += (first64_second64)*2;
    offset += first32_second32;
    offset += (i/16)*8;
    uint32_t toWrite = i + (((upper_lower)+1) << 28);

    printf("%d : %d %d %d %d %d %08X\n",i,upper_lower, first128_second128, first64_second64, first32_second32, offset, toWrite);
    if (upper_lower) {
      toWrite += (0x100000);
      bank1_ptr[offset] = toWrite;
      toWrite += (0x200000);
      bank3_ptr[offset] = toWrite;
      toWrite += (0x400000);
      bank5_ptr[offset] = toWrite;
      toWrite += (0x800000);
      bank7_ptr[offset] = toWrite;
    }
    else {
      toWrite += (0x100000);
      bank0_ptr[offset] = toWrite;
      toWrite += (0x200000);
      bank2_ptr[offset] = toWrite;
      toWrite += (0x400000);
      bank4_ptr[offset] = toWrite;
      toWrite += (0x800000);
      bank6_ptr[offset] = toWrite;
    }
  }
  
  for (int i=0;i<16;i++) {
    uint32_t word0 = bank0_ptr[i];
    uint32_t word1 = bank1_ptr[i];

    printf("%x %08X %08X\r\n", i, word0, word1);
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
  
  for (int stream=0; stream<4; stream++) {
    // globally bypass headers
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

    static l2_dma_cmd_t cmd;
    cmd.select = 7;
    cmd.length = 0;
    cmd.uram_addr = 1;
    cmd.id = 0;

    air_packet_l2_dma(pkt, stream, cmd);

    air_queue_dispatch_and_wait(q, wr_idx, pkt);
  }

  //
  // send the data
  //

  for (int stream=0; stream<4; stream++) {
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

    static l2_dma_cmd_t cmd;
    cmd.select = 0;
    cmd.length = 4;
    cmd.uram_addr = 0;
    cmd.id = stream;

    air_packet_l2_dma(pkt, stream, cmd);

    signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
    if (stream == 3) {
      air_queue_dispatch_and_wait(q, wr_idx, pkt);
    }
  }

  sleep(1);
  mlir_aie_print_dma_status(xaie, 7, 2);
  mlir_aie_print_dma_status(xaie, 8, 2);
  mlir_aie_print_dma_status(xaie, 9, 2);
  mlir_aie_print_dma_status(xaie, 10, 2);

  printf("\nChecking the output...\n");

  uint32_t errs = 0;
  for (int i=0; i<32; i++) {
    uint32_t d;
    if (i<16)
      d = mlir_aie_read_buffer_a(xaie, i) - 0x100000;
    else 
      d = mlir_aie_read_buffer_b(xaie, i-16) - 0x100000;
    if ((d & 0x0fffffff) != (i)) {
      printf("[7] Word %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }
  for (int i=0; i<32; i++) {
    uint32_t d;
    if (i<16)
      d = mlir_aie_read_buffer_c(xaie, i) - 0x300000;
    else 
      d = mlir_aie_read_buffer_d(xaie, i-16) - 0x300000;
    if ((d & 0x0fffffff) != (i)) {
      printf("[8] Word %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }
  for (int i=0; i<32; i++) {
    uint32_t d;
    if (i<16)
      d = mlir_aie_read_buffer_e(xaie, i) - 0x700000;
    else 
      d = mlir_aie_read_buffer_f(xaie, i-16) - 0x700000;
    if ((d & 0x0fffffff) != (i)) {
      printf("[9] Word %i : Expect %d, got %08X\n",i, i, d);
      errs++;
    }
  }
  for (int i=0; i<32; i++) {
    uint32_t d;
    if (i<16)
      d = mlir_aie_read_buffer_g(xaie, i) - 0xf00000;
    else 
      d = mlir_aie_read_buffer_i(xaie, i-16) - 0xf00000;
    if ((d & 0x0fffffff) != (i)) {
      printf("[A] Word %i : Expect %d, got %08X\n",i, i, d);
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
