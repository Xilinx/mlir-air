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

#include <air_host.h>
#include <air_tensor.h>

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

//
// global q ptr
//
queue_t *q = nullptr;

}

namespace air::herds::herd_0 {

int32_t mlir_read_buffer_buf0(int index);
int32_t mlir_read_buffer_buf1(int index);
int32_t mlir_read_buffer_buf2(int index);
int32_t mlir_read_buffer_buf3(int index);
int32_t mlir_read_buffer_buf4(int index);
int32_t mlir_read_buffer_buf5(int index);
int32_t mlir_read_buffer_buf6(int index);
int32_t mlir_read_buffer_buf7(int index);
void mlir_write_buffer_buf0(int index, int32_t value);
void mlir_write_buffer_buf1(int index, int32_t value);
void mlir_write_buffer_buf2(int index, int32_t value);
void mlir_write_buffer_buf3(int index, int32_t value);
void mlir_write_buffer_buf4(int index, int32_t value);
void mlir_write_buffer_buf5(int index, int32_t value);
void mlir_write_buffer_buf6(int index, int32_t value);
void mlir_write_buffer_buf7(int index, int32_t value);
}
using namespace air::herds::herd_0;

int
main(int argc, char *argv[])
{
  uint64_t row = 0;
  uint64_t col = 7;

  xaie = air_init_libxaie1();

  // Stomp
  for (int i=0; i<32*32; i++) {
    mlir_write_buffer_buf0(i, 0x0decaf);
    mlir_write_buffer_buf1(i, 0x1decaf);
    mlir_write_buffer_buf2(i, 0x2decaf);
    mlir_write_buffer_buf3(i, 0x3decaf);
    mlir_write_buffer_buf4(i, 0x4decaf);
    mlir_write_buffer_buf5(i, 0x5decaf);
    mlir_write_buffer_buf6(i, 0x6decaf);
    mlir_write_buffer_buf7(i, 0x7decaf);
  }

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

  tensor_t<int32_t,2> input;
  tensor_t<int32_t,2> output;

  input.shape[0] = input.shape[1] = 256;
  input.d = input.aligned = (int32_t*)malloc(sizeof(int32_t)*input.shape[0]*input.shape[1]);

  output.shape[0] = output.shape[1] = 256;
  output.d = output.aligned = (int32_t*)malloc(sizeof(int32_t)*output.shape[0]*output.shape[1]);

  auto handle = air_module_load_from_file(nullptr);
  assert(handle && "failed to open linked air module");

  auto graph_fn = (void (*)(void*,void*))dlsym((void*)handle, "_mlir_ciface_myAddOne");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in add_one.so");

  for (int i=0; i<input.shape[0]*input.shape[1]; i++) {
    input.d[i] = i;
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
    printf("before %d [7][2] : %08X -> %08X, [8][2] :%08X -> %08X, [7][3] : %08X -> %08X, [8][3] :%08X -> %08X\n", i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7);
  }

  void *i,*o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  for (int i=0; i<16; i++) { 
    int32_t rb0 = mlir_read_buffer_buf0(i);
    int32_t rb1 = mlir_read_buffer_buf1(i);
    int32_t rb2 = mlir_read_buffer_buf2(i);
    int32_t rb3 = mlir_read_buffer_buf3(i);
    int32_t rb4 = mlir_read_buffer_buf4(i);
    int32_t rb5 = mlir_read_buffer_buf5(i);
    int32_t rb6 = mlir_read_buffer_buf6(i);
    int32_t rb7 = mlir_read_buffer_buf7(i);
    printf(" after %d [7][2] : %08X -> %08X, [8][2] :%08X -> %08X, [7][3] : %08X -> %08X, [8][3] :%08X -> %08X\n", i, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7);
  }

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
    printf("fail %ld/%ld.\n", ((output.shape[0]*output.shape[1])-errors),
           output.shape[0]*output.shape[1]);
  }
}
