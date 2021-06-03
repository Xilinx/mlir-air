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

#include <sys/time.h>

#define SHMEM_BASE 0x020100000000LL

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#define VERBOSE 1
#define PROFILE 1

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie.inc"

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

//
// global q ptr
//
queue_t *q = nullptr;
uint32_t *bram_ptr;

}

static air_herd_shim_desc_t *_SD;

int64_t shim_location_data(air_herd_shim_desc_t *sd, int i, int j, int k) {
  return sd->location_data[i*16 + j*4 +k];
}

int64_t shim_channel_data(air_herd_shim_desc_t *sd, int i, int j, int k) {
  return sd->channel_data[i*16 + j*4 +k];
}

extern "C" {

void _mlir_ciface_air_shim_memcpy2d(uint32_t id, uint64_t x, uint64_t y, void* t,
                                    uint64_t offset_y, uint64_t offset_x,
                                    uint64_t length, uint64_t stride, uint64_t elem_per_stride) {

  auto shim_desc = _SD;//_air_host_active_herd->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id-1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id-1, x, y);

  tensor_t<uint32_t,2> *tt = (tensor_t<uint32_t,2> *)t;

  if (VERBOSE)
    printf("Do transfer %p with id %ld of length %ld on behalf of x=%ld, y=%ld shim col %ld channel %ld, offset %ld,%ld, stride %ld, elem %ld\n",
           tt->d, id, length, x, y, shim_col, shim_chan, offset_y, offset_x, stride, elem_per_stride);


  uint64_t addr = (u64)AIR_VCK190_SHMEM_BASE+0x4000;
  uint32_t *bounce_buffer = bram_ptr; //_air_host_bram_ptr;
  bool isMM2S = shim_chan >= 2;

  XAieDma_Shim dmaInst;
  XAieDma_ShimInitialize(&(TileInst[shim_col][0]), &dmaInst);

  if (isMM2S) {
    uint32_t *data_ptr = tt->d + (offset_y * tt->shape[1] + offset_x);
    uint32_t *bounce_ptr = bounce_buffer;
    for (int n=0; n<length; n+=elem_per_stride) {
      // This is the input, so we need to take what is in t and put it into the BRAM
      memcpy(bounce_ptr, data_ptr, elem_per_stride*sizeof(uint32_t));
      data_ptr += stride;
      bounce_ptr += elem_per_stride;
    }
  }

  //for (int n=0; n<length; n+=elem_per_stride) {
    uint32_t bd = shim_chan+1;
    auto burstlen = 4;
    XAieDma_ShimBdSetAddr(&dmaInst, bd, HIGH_ADDR(addr), LOW_ADDR(addr), sizeof(uint32_t) * length);
    XAieDma_ShimBdSetAxi(&dmaInst, bd, 0, burstlen, 0, 0, XAIE_ENABLE);
    XAieDma_ShimBdWrite(&dmaInst, bd);
    XAieDma_ShimSetStartBd((&dmaInst), shim_chan, bd); 

    XAieDma_ShimChControl((&dmaInst), shim_chan, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

    int count = 0;
    while (XAieDma_ShimPendingBdCount(&dmaInst, shim_chan)) {
      XAieLib_usleep(1000);
      count++;
      if (!(count % 1000)) {
        printf("%d seconds\n",count/1000);
        if (count == 2000) break;
      }
    }
    //addr += elem_per_stride;
  //}

  if (!isMM2S) {
    uint32_t *data_ptr = tt->d + (offset_y * tt->shape[1] + offset_x);
    uint32_t *bounce_ptr = bounce_buffer;
    for (int n=0; n<length; n+=elem_per_stride) {
      // This is the input, so we need to take what is in t and put it into the BRAM
      memcpy(data_ptr, bounce_ptr, elem_per_stride*sizeof(uint32_t));
      data_ptr += stride;
      bounce_ptr += elem_per_stride;
    }
  }
}

}

template<typename T>
void mm_out(tensor_t<T,2> *a, tensor_t<T,2> *b, tensor_t<T,2> *r)
{
  size_t a_h = a->shape[0];
  size_t a_w = a->shape[1];
  size_t b_h = b->shape[0];
  size_t b_w = b->shape[1];
  assert(a_w == b_h);

  for (size_t i=0; i<a_h; i++) {
    for (size_t j=0; j<b_w; j++) {
      size_t idx = i*b_w + j;
      r->d[idx] = (T)(0);
      for (size_t k=0, ke=a_w; k<a_w; k++) {
        T _a = a->d[i*a_w + k];
        T _b = b->d[k*b_w + j];
        r->d[idx] += _a * _b;
      }
    }
  }
}

int
main(int argc, char *argv[])
{
  uint64_t col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  if (VERBOSE)
    printCoreStatus(7,2);

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE+0x4000);
    for (int i=0; i<32*32; i++) {
      bram_ptr[i] = 0xdeface;
    }
  } else {
    printf("failed to open /dev/mem, exiting\n");
    exit(EXIT_FAILURE);
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

  tensor_t<uint32_t,2> input_A;
  tensor_t<uint32_t,2> input_B;
  tensor_t<uint32_t,2> output;
  tensor_t<uint32_t,2> output_ref;

  input_A.shape[0] = input_A.shape[1] = 64;
  input_A.d = input_A.aligned = (uint32_t*)malloc(sizeof(uint32_t)*input_A.shape[0]*input_A.shape[1]);

  input_B.shape[0] = input_B.shape[1] = 64;
  input_B.d = input_B.aligned = (uint32_t*)malloc(sizeof(uint32_t)*input_B.shape[0]*input_B.shape[1]);

  output.shape[0] = output.shape[1] = 64;
  output.d = output.aligned = (uint32_t*)malloc(sizeof(uint32_t)*output.shape[0]*output.shape[1]);

  output_ref.shape[0] = output_ref.shape[1] = 64;
  output_ref.d = output_ref.aligned = (uint32_t*)malloc(sizeof(uint32_t)*output_ref.shape[0]*output_ref.shape[1]);
  
  printf("%p %p %p\n", input_A.d, input_B.d, output.d);
  printf("loading aie_ctrl.so\n");
  auto handle = air_herd_load_from_file("./aie_ctrl.so");
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void *,void*))dlsym((void*)handle, "_mlir_ciface_task");
  assert(graph_fn && "failed to locate _mlir_ciface_task in .so");

  auto herd_desc = air_herd_get_desc(handle);
  _SD = herd_desc->shim_desc;

  for (int i=0; i<input_A.shape[0]*input_A.shape[1]; i++) {
    input_A.d[i] = (uint32_t)i;
    input_B.d[i] = (uint32_t)i+1.1f;
    output.d[i] = 0.0f;
    output_ref.d[i] = 0.0f;
  }

  mm_out(&input_A, &input_B, &output_ref);

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

  if (VERBOSE)
    printCoreStatus(7,2);

  for (int i=0; i<64; i++) {
    //printf("%d\n", mlir_read_buffer_buf0(i));
    //printf("%d\n", mlir_read_buffer_buf1(i));
    //printf("%d\n", mlir_read_buffer_buf2(i));
  }

  int errors = 0;
  auto output_size = output.shape[0]*output.shape[1];
  for (int i=0; i<output_size; i++) {
    float d = output.d[i];
    float ref = output_ref.d[i];
    if (d != ref) {
      errors++;
      printf("%04X: mismatch %f != %f\n", i, d, ref);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %ld/%ld.\n", (output_size-errors), output_size);
  }
}
