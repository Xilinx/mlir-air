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

#include "acdc_queue.h"
#include "hsa_defs.h"

#define SHMEM_BASE 0x020100000000LL

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie_inc.cpp"

template<typename T, int N>
struct tensor_t {
  T *d;
  T *aligned;
  size_t offset;
  size_t shape[N];
  size_t stride[N];

  size_t index(size_t n, size_t channel, size_t row, size_t col) const {
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    size_t idx = n * height * width * channels + channel * height * width + row * width + col;
    if (idx >= shape[0]*shape[1]*shape[2]*shape[3]) {
      printf("warning\n");
      return 0;
    }
    return idx;
  }

  tensor_t() {
    d = aligned = nullptr;
    offset = 0;
    for (int i=0; i<N; i++)
      shape[i] = stride[i] = 0;
  }
};


void printCoreStatus(int col, int row) {
  u32 status, coreTimerLow, locks, PC;
	status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x032004);
	coreTimerLow = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0340F8);
	locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
	PC = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x00030280);
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
    dma_s2mm_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x01D160);
    dma_mm2s_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x01D164);
  }
  else {
    dma_s2mm_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x01DF00);
    dma_mm2s_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x01DF10);
  }
  printf((row == 0) ? "SHIM " : "TILE ");
  printf("DMA [%d, %d] s2mm status is %08X, mm2s status %08X\n",col, row, dma_s2mm_status, dma_mm2s_status);

}

hsa_status_t queue_create(uint32_t size, uint32_t type, queue_t **queue)
{
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  uint64_t *bram_ptr = (uint64_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, SHMEM_BASE);

  printf("Opened shared memory paddr: %p vaddr: %p\n", (void*)SHMEM_BASE, (void*)bram_ptr);
  uint64_t q_paddr = bram_ptr[0];
  uint64_t q_offset = q_paddr - SHMEM_BASE;
  queue_t *q = (queue_t*)( ((size_t)bram_ptr) + q_offset );
  printf("Queue location at paddr: %p vaddr: %p\n", (void*)bram_ptr[0], (void*)q);

  if (q->id !=  0xacdc) {
    printf("%s error invalid id %x\n", __func__, (unsigned)q->id);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->size != size) {
    printf("%s error size mismatch %d\n", __func__, (unsigned)q->size);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->type != type) {
    printf("%s error type mismatch %d\n", __func__, q->type);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  uint64_t base_address_offset = q->base_address - SHMEM_BASE;
  q->base_address_vaddr = ((size_t)bram_ptr) + base_address_offset;

  q->doorbell = 0xffffffffffffffffUL;
  q->last_doorbell = 0;

  *queue = q;
  return HSA_STATUS_SUCCESS;
}

//
// global q ptr
//
queue_t *q = nullptr;
uint32_t *bram_ptr;

}

#define BRAM_ADDR 0x4000+0x020100000000LL
#define DMA_COUNT 64

extern "C" {

long TILE_TO_SHIM_DMA[3][2][2] {0};
long ARG_TO_SHIM_DMA_CHANNEL[3][2][2] {0};
// Let's imagine the compiler made this function
void build_tile_to_shim_dma_mapping() {
  TILE_TO_SHIM_DMA[0][0][0] = 2L;
  TILE_TO_SHIM_DMA[1][0][0] = 2L;
  TILE_TO_SHIM_DMA[2][0][0] = 2L;

  TILE_TO_SHIM_DMA[0][1][0] = 3L;
  TILE_TO_SHIM_DMA[1][1][0] = 3L;
  TILE_TO_SHIM_DMA[2][1][0] = 2L;

  TILE_TO_SHIM_DMA[0][0][1] = 6L;
  TILE_TO_SHIM_DMA[1][0][1] = 6L;
  TILE_TO_SHIM_DMA[2][0][1] = 3L;
  TILE_TO_SHIM_DMA[0][1][1] = 7L;
  TILE_TO_SHIM_DMA[1][1][1] = 7L;
  TILE_TO_SHIM_DMA[2][1][1] = 3L;
}

void build_arg_to_shim_dma_channel_mapping() {
  ARG_TO_SHIM_DMA_CHANNEL[0][0][0] = XAIEDMA_SHIM_CHNUM_MM2S0;
  ARG_TO_SHIM_DMA_CHANNEL[1][0][0] = XAIEDMA_SHIM_CHNUM_MM2S1;
  ARG_TO_SHIM_DMA_CHANNEL[2][0][0] = XAIEDMA_SHIM_CHNUM_S2MM0;

  ARG_TO_SHIM_DMA_CHANNEL[0][1][0] = XAIEDMA_SHIM_CHNUM_MM2S0;
  ARG_TO_SHIM_DMA_CHANNEL[1][1][0] = XAIEDMA_SHIM_CHNUM_MM2S1;
  ARG_TO_SHIM_DMA_CHANNEL[2][1][0] = XAIEDMA_SHIM_CHNUM_S2MM1;

  ARG_TO_SHIM_DMA_CHANNEL[0][0][1] = XAIEDMA_SHIM_CHNUM_MM2S0;
  ARG_TO_SHIM_DMA_CHANNEL[1][0][1] = XAIEDMA_SHIM_CHNUM_MM2S1;
  ARG_TO_SHIM_DMA_CHANNEL[2][0][1] = XAIEDMA_SHIM_CHNUM_S2MM0;

  ARG_TO_SHIM_DMA_CHANNEL[0][1][1] = XAIEDMA_SHIM_CHNUM_MM2S0;
  ARG_TO_SHIM_DMA_CHANNEL[1][1][1] = XAIEDMA_SHIM_CHNUM_MM2S1;
  ARG_TO_SHIM_DMA_CHANNEL[2][1][1] = XAIEDMA_SHIM_CHNUM_S2MM1;
}

void printMems(int i, char *prefix) {
    int32_t rb0 = mlir_read_buffer_buf0(i);
    int32_t rb1 = mlir_read_buffer_buf1(i);
    int32_t rb2 = mlir_read_buffer_buf2(i);
    int32_t rb3 = mlir_read_buffer_buf3(i);
    int32_t rb4 = mlir_read_buffer_buf4(i);
    int32_t rb5 = mlir_read_buffer_buf5(i);
    int32_t rb6 = mlir_read_buffer_buf6(i);
    int32_t rb7 = mlir_read_buffer_buf7(i);
    int32_t rb8 = mlir_read_buffer_buf8(i);
    int32_t rb9 = mlir_read_buffer_buf9(i);
    int32_t rb10 = mlir_read_buffer_buf10(i);
    int32_t rb11 = mlir_read_buffer_buf11(i);
    printf("%s %d [7][2] : %08X * %08X -> %08X, [8][2] :%08X * %08X -> %08X, [7][3] : %08X * %08X -> %08X, [8][3] :%08X * %08X-> %08X\n", prefix, i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8, rb9, rb10, rb11);
}


void _mlir_ciface_air_shim_memcpy(uint32_t id, uint64_t x, uint64_t y, void* t, uint64_t offset, uint64_t length) {
  printf("Do transfer with id %ld of length %ld on behalf of x=%ld, y=%ld using shim DMA %ld channel %ld, offset is %ld\n", id, length, x, y, TILE_TO_SHIM_DMA[id-1][x][y], ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y], offset);

  tensor_t<int32_t,1> *tt = (tensor_t<int32_t,1> *)t;

  // Used to use BRAM_ADDR + 0x4000 as the data address
  uint64_t addr = (u64)BRAM_ADDR;
  int32_t *bounce_buffer = (int32_t *)bram_ptr;
  if (id < 3) {
    // This are the inputs, so we need to take what is in t and put it into the BRAM
    for (int i=0; i<length; i++) {
      bounce_buffer[i] = tt->d[offset + i];
    }
  }

  printDMAStatus(TILE_TO_SHIM_DMA[id-1][x][y], 0);
  printDMAStatus(x+7, y+2);
  printCoreStatus(x+7, y+2);

  printMems(2, "before"); 

  auto burstlen = 4;
  XAieDma_Shim ShimDmaInst1;
  XAieDma_ShimSoftInitialize(&(TileInst[TILE_TO_SHIM_DMA[id-1][x][y]][0]), &ShimDmaInst1);
  u8 bd = 1+ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y];
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, bd, HIGH_ADDR(addr), LOW_ADDR(addr), length*sizeof(uint32_t));
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, bd , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, bd);             // We don't really care, we just want these to be unique and none zero
  XAieDma_ShimSetStartBd((&ShimDmaInst1), ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y], bd);

  auto ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y]);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  XAieDma_ShimChControl((&ShimDmaInst1), ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y], XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, ARG_TO_SHIM_DMA_CHANNEL[id-1][x][y])) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }
  if (id == 3) {
    // This is the output, so we need to take what is in t and put it into the BRAM
    printf("Copy %ld samples to the output starting at %ld\n",length, offset);
    for (int i=0; i<length; i++) {
      tt->d[offset + i] = bounce_buffer[i];
    }
  }


  printDMAStatus(TILE_TO_SHIM_DMA[id-1][x][y], 0);
  printDMAStatus(x+7, y+2);
  printCoreStatus(x+7, y+2);
  printMems(2, "after"); 
}



}



int
main(int argc, char *argv[])
{
  uint64_t row = 0;
  uint64_t col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  printCoreStatus(7,2);


  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  build_tile_to_shim_dma_mapping();
  build_arg_to_shim_dma_channel_mapping();

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<32*32; i++) {
      bram_ptr[i] = 0xdeface;
    }
  }

  // Stomp
  for (int i=0; i<DMA_COUNT*4*4; i++) {
    mlir_write_buffer_buf0(i, 0x0decaf);
    mlir_write_buffer_buf1(i, 0x1decaf);
    mlir_write_buffer_buf2(i, 0x2decaf);
    mlir_write_buffer_buf3(i, 0x3decaf);
    mlir_write_buffer_buf4(i, 0x4decaf);
    mlir_write_buffer_buf5(i, 0x5decaf);
    mlir_write_buffer_buf6(i, 0x6decaf);
    mlir_write_buffer_buf7(i, 0x7decaf);
    mlir_write_buffer_buf8(i, 0x8decaf);
    mlir_write_buffer_buf9(i, 0x9decaf);
    mlir_write_buffer_buf10(i, 0xadecaf);
    mlir_write_buffer_buf11(i, 0xbdecaf);
  }

  printCoreStatus(7,2);
  // create the queue
  auto ret = queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q);
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

  tensor_t<int32_t,1> input_A;
  tensor_t<int32_t,1> input_B;
  tensor_t<int32_t,1> output;

  input_A.shape[0] = DMA_COUNT*4*4;
  input_A.d = input_A.aligned = (int32_t*)malloc(sizeof(int32_t)*input_A.shape[0]);

  input_B.shape[0] = DMA_COUNT*4*4;
  input_B.d = input_B.aligned = (int32_t*)malloc(sizeof(int32_t)*input_B.shape[0]);

  output.shape[0] = DMA_COUNT*4*4;
  output.d = output.aligned = (int32_t*)malloc(sizeof(int32_t)*output.shape[0]);
  
  printf("loading aie_ctrl.so\n");
  auto handle = dlopen("./aie_ctrl.so", RTLD_NOW);
  if (!handle) {
    printf("%s\n",dlerror());
  }
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void *,void*))dlsym(handle, "_mlir_ciface_task");
  assert(graph_fn && "failed to locate _mlir_ciface_task in vecmul.so");

  for (int i=0; i<input_A.shape[0]; i++) {
    input_A.d[i] = i;
    input_B.d[i] = i+1;
  }
  for (int i=0; i<16; i++) { 
    int32_t rb0 = mlir_read_buffer_buf0(i);
    int32_t rb1 = mlir_read_buffer_buf1(i);
    int32_t rb2 = mlir_read_buffer_buf2(i);
    int32_t rb3 = mlir_read_buffer_buf3(i);
    int32_t rb4 = mlir_read_buffer_buf4(i);
    int32_t rb5 = mlir_read_buffer_buf5(i);
    int32_t rb6 = mlir_read_buffer_buf6(i);
    int32_t rb7 = mlir_read_buffer_buf7(i);
    int32_t rb8 = mlir_read_buffer_buf8(i);
    int32_t rb9 = mlir_read_buffer_buf9(i);
    int32_t rb10 = mlir_read_buffer_buf10(i);
    int32_t rb11 = mlir_read_buffer_buf11(i);
    printf("before %d [7][2] : %08X * %08X -> %08X, [8][2] :%08X * %08X -> %08X, [7][3] : %08X * %08X -> %08X, [8][3] :%08X * %08X-> %08X\n", i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8, rb9, rb10, rb11);
  }

  void *a, *b,*o;
  a = &input_A;
  b = &input_B;
  o = &output;
  graph_fn(a, b, o);

  for (int i=0; i<16; i++) { 
    int32_t rb0 = mlir_read_buffer_buf0(i);
    int32_t rb1 = mlir_read_buffer_buf1(i);
    int32_t rb2 = mlir_read_buffer_buf2(i);
    int32_t rb3 = mlir_read_buffer_buf3(i);
    int32_t rb4 = mlir_read_buffer_buf4(i);
    int32_t rb5 = mlir_read_buffer_buf5(i);
    int32_t rb6 = mlir_read_buffer_buf6(i);
    int32_t rb7 = mlir_read_buffer_buf7(i);
    int32_t rb8 = mlir_read_buffer_buf8(i);
    int32_t rb9 = mlir_read_buffer_buf9(i);
    int32_t rb10 = mlir_read_buffer_buf10(i);
    int32_t rb11 = mlir_read_buffer_buf11(i);
    printf(" after %d [7][2] : %08X * %08X -> %08X, [8][2] :%08X * %08X -> %08X, [7][3] : %08X * %08X -> %08X, [8][3] :%08X * %08X-> %08X\n", i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8, rb9, rb10, rb11);
  }

  printCoreStatus(7,2);

  int errors = 0;
  for (int i=0; i<output.shape[0]; i++) {
    uint32_t a = input_A.d[i];
    uint32_t b = input_B.d[i];
    uint32_t d = output.d[i];
    if (d != (a*b)) {
      errors++;
      printf("%04X: mismatch %x != %x * %x\n", i, d, a, b);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %ld/%ld.\n", (output.shape[0]-errors), output.shape[0]);
  }
}
