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

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie_inc.cpp"
#undef TileInst
#undef TileDMAInst

//
// global q ptr
//
queue_t *q = nullptr;
float *bram_ptr;

}

void printCoreStatus(int col, int row) {
  u32 status, coreTimerLow, locks, PC;
	status = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x032004);
	coreTimerLow = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x0340F8);
	locks = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x0001EF00);
	PC = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x00030280);
	printf("Core [%d, %d] @ %04X: status is %08X, timer is %u, locks are %08X\n",col, row, PC, status, coreTimerLow, locks);
	for (int lock=0;lock<16;lock++) {
		u32 two_bits = (locks >> (lock*2)) & 0x3;
		if (two_bits) {
			printf("Lock %d: ", lock);
			u32 acquired = two_bits & 0x1;
			u32 value = two_bits & 0x2;
			if (acquired)
				printf("Acquired ");
			printf(value?"1":"0");
			printf("\n");
		}
	}
}

void printDMAStatus(int col, int row) {
  u32 dma_s2mm_status, dma_mm2s_status;
  if (row == 0) {
    dma_s2mm_status = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x01D160);
    dma_mm2s_status = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x01D164);
  }
  else {
    dma_s2mm_status = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x01DF00);
    dma_mm2s_status = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x01DF10);
  }
  printf((row == 0) ? "SHIM " : "TILE ");
  printf("DMA [%d, %d] s2mm status is %08X, mm2s status %08X\n",col, row, dma_s2mm_status, dma_mm2s_status);

}

#define GRID_SIZE 4096

void printMems(int i, const char *prefix) {
    float rb0 = mlir_read_buffer_buf0(i);
    float rb1 = mlir_read_buffer_buf1(i);
    float rb2 = mlir_read_buffer_buf2(i);
    float rb3 = mlir_read_buffer_buf3(i);
    float rb4 = mlir_read_buffer_buf4(i);
    float rb5 = mlir_read_buffer_buf5(i);
    float rb6 = mlir_read_buffer_buf6(i);
    float rb7 = mlir_read_buffer_buf7(i);
    float rb8 = mlir_read_buffer_buf8(i);
    float rb9 = mlir_read_buffer_buf9(i);
    float rb10 = mlir_read_buffer_buf10(i);
    float rb11 = mlir_read_buffer_buf11(i);
    printf("%s %d [7][2] : %f * %f -> %f, [8][2] :%f * %f -> %f, [7][3] : %f * %f -> %f, [8][3] :%f * %f-> %f\n", prefix, i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8, rb9, rb10, rb11);
}

int
main(int argc, char *argv[])
{
  uint64_t col = 7;

  xaie = air_init_libxaie1();

  if (VERBOSE)
    printCoreStatus(7,2);


  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (float *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE+0x4000);
    for (int i=0; i<32*32; i++) {
      bram_ptr[i] = 0.0;
    }
  }

  // Stomp
  for (int i=0; i<GRID_SIZE; i++) {
    mlir_write_buffer_buf0(i, 1.0f);
    mlir_write_buffer_buf1(i, 2.0f);
    mlir_write_buffer_buf2(i, 3.0f);
    mlir_write_buffer_buf3(i, 4.0f);
    mlir_write_buffer_buf4(i, 5.0f);
    mlir_write_buffer_buf5(i, 6.0f);
    mlir_write_buffer_buf6(i, 7.0f);
    mlir_write_buffer_buf7(i, 8.0f);
    mlir_write_buffer_buf8(i, 9.0f);
    mlir_write_buffer_buf9(i, 10.0f);
    mlir_write_buffer_buf10(i, 11.0f);
    mlir_write_buffer_buf11(i, 12.0f);
  }

  if (VERBOSE) {
    for (int i=0; i<16; i++) {
      printMems(i, "before");
    }
  }

  if (VERBOSE)
    printCoreStatus(7,2);
  // create the queue
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(herd_pkt);
  herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  //
  // Set up a 1x3 herd starting 7,0
  //

  herd_pkt->arg[0]  = AIR_PKT_TYPE_HERD_INITIALIZE;
  herd_pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  herd_pkt->arg[0] |= (1L << 40);
  herd_pkt->arg[0] |= (7L << 32);
  herd_pkt->arg[0] |= (3L << 24);
  herd_pkt->arg[0] |= (0L << 16);
  
  herd_pkt->arg[1] = 0;  // Herd ID 0
  herd_pkt->arg[2] = 0;
  herd_pkt->arg[3] = 0;

  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&herd_pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  //signal_store_release((signal_t*)&q->doorbell, wr_idx);

  tensor_t<float,1> input_A;
  tensor_t<float,1> input_B;
  tensor_t<float,1> output;

  input_A.shape[0] = GRID_SIZE;
  input_A.d = input_A.aligned = (float*)malloc(sizeof(float)*input_A.shape[0]);

  input_B.shape[0] = GRID_SIZE;
  input_B.d = input_B.aligned = (float*)malloc(sizeof(float)*input_B.shape[0]);

  output.shape[0] = GRID_SIZE;
  output.d = output.aligned = (float*)malloc(sizeof(float)*output.shape[0]);

  printf("loading aie_ctrl.so\n");
  auto handle = air_herd_load_from_file("./aie_ctrl.so");
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void *,void*))dlsym((void*)handle, "_mlir_ciface_task");
  assert(graph_fn && "failed to locate _mlir_ciface_task in vecmul.so");

  for (int i=0; i<input_A.shape[0]; i++) {
    input_A.d[i] = 1.0f*i;
    input_B.d[i] = 2.0f*i;
  }

  if (VERBOSE) {
    for (int i=0; i<16; i++) {
      printMems(i, "before");
    }
  }

  void *a, *b,*o;
  a = &input_A;
  b = &input_B;
  o = &output;
  struct timeval before, after;
  long diff_s, diff_us;
  gettimeofday(&before, NULL);
  graph_fn(a, b, o);
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
      printMems(i, "after");
    }
    printCoreStatus(7,2);
  }

  int errors = 0;
  for (int i=0; i<output.shape[0]; i++) {
    float a = input_A.d[i];
    float b = input_B.d[i];
    float d = output.d[i];
    if (d != (a*b)) {
      errors++;
      printf("%04X: mismatch %f != %f * %f\n", i, d, a, b);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %ld/%ld.\n", (output.shape[0]-errors), output.shape[0]);
  }
}
