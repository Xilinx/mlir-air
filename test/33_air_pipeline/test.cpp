// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include <xaiengine.h>

#include "air_host.h"
#include "air_tensor.h"

int main(int argc, char *argv[]) {
  auto col = 3;
  auto row = 3;

  /*aie_libxaie_ctx_t *xaie = */ air_init_libxaie1();

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

#define DATA_LENGTH 1024
#define DATA_TYPE int

  tensor_t<DATA_TYPE, 1> input_a, input_b;
  tensor_t<DATA_TYPE, 1> output;
  input_a.shape[0] = DATA_LENGTH;
  input_a.alloc = input_a.data =
      (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * input_a.shape[0]);

  input_b.shape[0] = DATA_LENGTH;
  input_b.alloc = input_b.data =
      (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * input_b.shape[0]);

  output.shape[0] = input_a.shape[0];
  output.alloc = output.data =
      (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * output.shape[0]);
  DATA_TYPE d = 1;
  for (int i = 0; i < input_a.shape[0]; i++) {
    input_a.data[i] = d;
    input_b.data[i] = ((DATA_TYPE)DATA_LENGTH) + d;
    output.data[i] = -1;
    d += 1;
  }

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open aie_ctrl.so");

  auto launch = (void (*)(void *, void *, void *))dlsym((void *)handle,
                                                        "_mlir_ciface_launch");
  assert(launch && "failed to locate _mlir_ciface_launch in .so");

  launch((void *)&input_a, (void *)&input_b, (void *)&output);

  int errors = 0;

  for (int i = 0; i < DATA_LENGTH; i++) {
    DATA_TYPE ref = (input_a.data[i] * input_b.data[i]) + (DATA_TYPE)1 +
                    (DATA_TYPE)2 + (DATA_TYPE)3;
    if (output.data[i] != ref) {
      printf("output[%d] = %d (expected %d)\n", i, output.data[i], ref);
      errors++;
    }
  }

  free(output.alloc);
  free(input_a.alloc);
  free(input_b.alloc);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", (DATA_LENGTH - errors), DATA_LENGTH);
    return -1;
  }
}
