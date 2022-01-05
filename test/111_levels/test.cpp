// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <cstdio>
#include <cassert>
#include <climits>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "acdc_queue.h"
#include "hsa_defs.h"
#include "air_host.h"

#include "aie_inc.cpp"

#define XFR_SIZE 512
#define XFR_BYTES XFR_SIZE*4

int main(int argc, char *argv[])
{
  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);
  
  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  for (int i=0; i<XFR_SIZE; i++) {
    mlir_aie_write_buffer_buf1(xaie,i,i+0xbeef1000);
    mlir_aie_write_buffer_buf2(xaie,i,i+0xbeef2000);
    mlir_aie_write_buffer_buf3(xaie,i,i+0xbeef3000);
    mlir_aie_write_buffer_buf4(xaie,i,i+0xbeef4000);
  }

  mlir_aie_print_dma_status(xaie, 7, 1);
  mlir_aie_print_dma_status(xaie, 7, 2);
  mlir_aie_print_dma_status(xaie, 7, 3);
  mlir_aie_print_dma_status(xaie, 7, 4);

  XAieGbl_Write32(xaie->TileInst[7][0].TileAddr + 0x00033008, 0xFF);

  uint32_t reg = XAieGbl_Read32(xaie->TileInst[7][0].TileAddr + 0x00033004);
  printf("REG %x\n", reg);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1) {
    printf("failed to open /dev/mem\n");
    return -1;
  }
  
  uint32_t *bank0_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE);
  uint32_t *bank1_ptr = (uint32_t *)mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_L2_DMA_BASE+0x20000);
  uint32_t *dram_ptr  = (uint32_t *)mmap(NULL, 0x100000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_DDR_BASE);
  for (int i=0; i<4*XFR_SIZE; i++) {
    bank0_ptr[i] = 0xACDC000 + i + (1 << 28);
    bank1_ptr[i] = 0xACDC000 + i + (2 << 28);
  }

  printf("L2 before: \n");
  for (int i=0; i<4*XFR_SIZE; i++) {
    printf("%4d contains %08x %08x\n",i,bank0_ptr[i],bank1_ptr[i]);
  }
  for (int i=0; i<5*XFR_SIZE; i++) {
    // 3D DMA address generation
    //           X Y Z
    // increment 1 2 8
    // wrap      2 4 max
    // offset    4 1 8
    int an = 4*((i/1)%2) + 1*((i/2)%4) + 8*((i/8)%UINT_MAX); 
    dram_ptr[i] = 0x100 + i;
  }
  printf("DRAM before: \n");
  for (int i=0; i<XFR_SIZE; i++) {
    printf("%4d contains %08x %08x %08x %08x %08x\n",i,dram_ptr[i],dram_ptr[i+XFR_SIZE],dram_ptr[i+2*XFR_SIZE],dram_ptr[i+3*XFR_SIZE],dram_ptr[i+4*XFR_SIZE]);
  }

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");
  
  //
  // Set up a 1x4 herd starting 7,1
  //
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(pkt, 0, 7, 1, 1, 4);
  air_queue_dispatch_and_wait(q, wr_idx, pkt);
  
  //  
  // enable headers
  //
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

  static l2_dma_cmd_t cmd;
  cmd.select = 7;
  cmd.length = 0;
  cmd.uram_addr = 0;
  cmd.id = 0;

  uint64_t stream = 0;
  air_packet_l2_dma(pkt, stream, cmd);

  //
  //  copy data from L3 to L2
  //
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_cdma_memcpy(pkt, uint64_t(AIR_VCK190_L2_DMA_BASE), uint64_t(AIR_VCK190_DDR_BASE), XFR_BYTES);

  //
  // send the data
  //
  int sel=2;
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

  cmd.select = sel;
  cmd.length = 128;
  cmd.uram_addr = 0;
  cmd.id = sel+1;

  air_packet_l2_dma(pkt, stream, cmd);

  //
  // read the data back
  //
  sel = 4;
  dispatch_packet_t *p[4];
  for (int i = 0; i < 4; i++) { 
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
    pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;

    cmd.select = sel;
    cmd.length = 128;
    cmd.uram_addr = 128*i;
    cmd.id = 0xA+i;

    air_packet_l2_dma(pkt, stream, cmd);
    signal_create(1, 0, NULL, (signal_t *)&pkt->completion_signal);
    p[i] = pkt;
  }

  air_queue_dispatch_and_wait(q, wr_idx, p[3]);
  air_queue_wait(q, p[0]);
  air_queue_wait(q, p[1]);
  air_queue_wait(q, p[2]);
  sleep(1);

  //
  //  copy data from L2 to L3
  //
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;
  pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_cdma_memcpy(pkt, uint64_t(AIR_VCK190_DDR_BASE+XFR_BYTES), uint64_t(AIR_VCK190_L2_DMA_BASE), XFR_BYTES*4);
 
  air_queue_dispatch_and_wait(q, wr_idx, pkt);

  printf("Tile after: \n");
  for (int i=0; i<XFR_SIZE; i++) {
    uint32_t d0 = mlir_aie_read_buffer_buf1(xaie,i);
    uint32_t d1 = mlir_aie_read_buffer_buf2(xaie,i);
    uint32_t d2 = mlir_aie_read_buffer_buf3(xaie,i);
    uint32_t d3 = mlir_aie_read_buffer_buf4(xaie,i);
    printf("%4d contains %08x %08x %08x %08x\n",i,d0,d1,d2,d3);
  }
  printf("L2 after: \n");
  for (int i=0; i<4*XFR_SIZE; i++) {
    printf("%4d contains %08x %08x\n",i,bank0_ptr[i],bank1_ptr[i]);
  }
  printf("DRAM after: \n");
  int errs = 0;
  for (int i=0; i<XFR_SIZE; i++) {
    if (!(dram_ptr[i] == dram_ptr[i+XFR_SIZE] && 
        dram_ptr[i+XFR_SIZE] == dram_ptr[i+2*XFR_SIZE] &&
        dram_ptr[i+2*XFR_SIZE] == dram_ptr[i+3*XFR_SIZE] &&
        dram_ptr[i+3*XFR_SIZE] == dram_ptr[i+4*XFR_SIZE])) errs++;
    printf("%4d contains %08x %08x %08x %08x %08x\n",i,dram_ptr[i],dram_ptr[i+XFR_SIZE],dram_ptr[i+2*XFR_SIZE],dram_ptr[i+3*XFR_SIZE],dram_ptr[i+4*XFR_SIZE]);
  }

  if (errs) {
    printf("fail.\n");
    return 1;
  }

  printf("PASS!\n");

  return 0;
}
