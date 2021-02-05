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

hsa_status_t queue_create(uint32_t size, uint32_t type, queue_t **queue)
{
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  uint64_t *bram_ptr = (uint64_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, SHMEM_BASE);

  printf("Opened shared memory paddr: %p vaddr: %p\n", SHMEM_BASE, bram_ptr);
  uint64_t q_paddr = bram_ptr[0];
  uint64_t q_offset = q_paddr - SHMEM_BASE;
  queue_t *q = (queue_t*)( ((size_t)bram_ptr) + q_offset );
  printf("Queue location at paddr: %p vaddr: %p\n", bram_ptr[0], q);

  if (q->id !=  0xacdc) {
    printf("%s error invalid id %x\n", __func__, q->id);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->size != size) {
    printf("%s error size mismatch %d\n", __func__, q->size);
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
#define DMA_COUNT 32

extern "C" {

void _mlir_ciface_acap_add_one_hw_kernel_AtenAcapOp_I64_I64() {

}

void _mlir_ciface_acap_L2_dma_copy_arg0(tensor_t<float,2> *input, tensor_t<float,2> *output, size_t dim1_idx, size_t dim0_idx) {
  printf("copy L2 arg0 %p %p %d %d\n", input->d, output->d, dim1_idx, dim0_idx);

  
  uint64_t row = 7;
  uint64_t col = 0;

  for (int row_offset=0; row_offset<DMA_COUNT; row_offset++) {
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = input->d[(row_offset+dim1_idx)*input->shape[0] + dim0_idx + i];
    }

    uint64_t wr_idx = queue_add_write_index(q, 1);
    uint64_t packet_id = wr_idx % q->size;

    dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    initialize_packet(pkt);
    pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
    pkt->arg[0] = AIR_PKT_TYPE_SHIM_DMA_MEMCPY;
    pkt->arg[0] |= (row << 16);
    pkt->arg[0] |= (col << 32);
    uint64_t flags = 0x1;
    pkt->arg[0] |= (flags << 48);

    uint32_t burst_len = 4;
    uint64_t direction = 1;
    uint64_t channel = XAIEDMA_SHIM_CHNUM_MM2S0;

    pkt->arg[1] = burst_len;
    pkt->arg[1] |= (direction << 32);
    pkt->arg[1] |= (channel << 48);
    pkt->arg[2] = BRAM_ADDR;
    pkt->arg[3] = DMA_COUNT*sizeof(float);

    signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
    signal_store_release((signal_t*)&q->doorbell, wr_idx);
    
    while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
      printf("packet completion signal timeout!\n");
      printf("%x\n", pkt->header);
      printf("%x\n", pkt->type);
      printf("%x\n", pkt->completion_signal);
    }
  }
}

void _mlir_ciface_acap_L2_dma_copy_arg1(tensor_t<float,2> *input, tensor_t<float,2> *output, size_t dim1_idx, size_t dim0_idx) {

  printf("copy L2 arg1 %p %p %d %d\n", input->d, output->d, dim1_idx, dim0_idx);
  for (int row_offset=0; row_offset<DMA_COUNT; row_offset++) {
    uint64_t wr_idx = queue_add_write_index(q, 1);
    uint64_t packet_id = wr_idx % q->size;
    uint64_t row = 7;
    uint64_t col = 0;

    dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    initialize_packet(pkt);
    pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;
    pkt->arg[0] = AIR_PKT_TYPE_SHIM_DMA_MEMCPY;
    pkt->arg[0] |= (row << 16);
    pkt->arg[0] |= (col << 32);
    uint64_t flags = 0x1;
    pkt->arg[0] |= (flags << 48);
    
    uint64_t direction = 0;
    uint64_t channel = XAIEDMA_SHIM_CHNUM_S2MM0;

    uint32_t burst_len = 4;
    pkt->arg[1] = burst_len;
    pkt->arg[1] |= (direction << 32);
    pkt->arg[1] |= (channel << 48);
    pkt->arg[2] = BRAM_ADDR+DMA_COUNT*sizeof(float);
    pkt->arg[3] = DMA_COUNT*sizeof(float);

    signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
    signal_store_release((signal_t*)&q->doorbell, wr_idx);

    while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
      printf("packet completion signal timeout!\n");
      printf("%x\n", pkt->header);
      printf("%x\n", pkt->type);
      printf("%x\n", pkt->completion_signal);
    }
    
    for (int i=0; i<DMA_COUNT; i++) {
      output->d[(row_offset+dim1_idx)*output->shape[0] + dim0_idx + i] = bram_ptr[DMA_COUNT+i];
    }
  }
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

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<32*32; i++) {
      bram_ptr[i] = 0xdeface;
    }
  }

  for (int i=0; i<16*2; i++) {
    XAieTile_DmWriteWord(&(TileInst[col][2]), 0x1000+i*4, 0xdecaf);
  }

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

  tensor_t<float,2> input;
  tensor_t<float,2> output;

  input.shape[0] = 256;
  input.shape[1] = 256;
  input.d = input.aligned = (float*)malloc(sizeof(float)*input.shape[0]*input.shape[1]);

  output.shape[0] = 256;
  output.shape[1] = 256;
  output.d = output.aligned = (float*)malloc(sizeof(float)*output.shape[0]*output.shape[1]);
  
  printf("loading add_one.so\n");
  auto handle = dlopen("./add_one.so", RTLD_NOW);
  if (!handle) {
    printf("%s\n",dlerror());
  }
  assert(handle && "failed to open add_one.so");

  auto graph_fn = (void (*)(void*,void*))dlsym(handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in add_one.so");

  for (int i=0; i<input.shape[0]*input.shape[1]; i++) {
    input.d[i] = (float)i;
    output.d[i] = -1.11111f;
  }

  void *i,*o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  int errors = 0;
  for (int i=0; i<output.shape[0]*output.shape[1]; i++) {
    uint32_t d = output.d[i];
    if (d != (i+1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, i);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT-errors), DMA_COUNT);
  }

}