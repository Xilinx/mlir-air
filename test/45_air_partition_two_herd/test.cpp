// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

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

#define DATA_SIZE 10240

int main(int argc, char *argv[]) {
  queue_t *q = nullptr;

  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                              AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  tensor_t<uint32_t, 1> input_A;
  tensor_t<uint32_t, 1> input_B;
  tensor_t<uint32_t, 1> input_C;
  tensor_t<uint32_t, 1> output;
  tensor_t<uint32_t, 1> output_ref;

  input_A.shape[0] = DATA_SIZE;
  input_A.alloc = input_A.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input_A.shape[0]);

  input_B.shape[0] = DATA_SIZE;
  input_B.alloc = input_B.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input_B.shape[0]);

  input_C.shape[0] = DATA_SIZE;
  input_C.alloc = input_C.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input_C.shape[0]);

  output.shape[0] = DATA_SIZE;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0]);

  output_ref.shape[0] = DATA_SIZE;
  output_ref.alloc = output_ref.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output_ref.shape[0]);

  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open linked air module");

  auto herd_fn = (void (*)(void *, void *, void *, void *))dlsym(
      (void *)handle, "_mlir_ciface_forward");
  assert(herd_fn && "failed to locate _mlir_ciface_forward in .so");

  for (int i = 0; i < input_A.shape[0]; i++) {
    input_A.data[i] = (rand() % 1024) + 1;
    input_B.data[i] = (rand() % 1024) + 1;
    input_C.data[i] = (rand() % 1024) + 1;
    output.data[i] = 0xdeadcafe;
    output_ref.data[i] = input_A.data[i] * (input_B.data[i] + input_C.data[i]);
  }

  void *a, *b, *c, *o;
  a = &input_A;
  b = &input_B;
  c = &input_C;
  o = &output;

  // run it
  herd_fn(a, b, c, o);

  int errors = 0;
  auto output_size = output.shape[0];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref.data[i];
    if (d != ref) {
      errors++;
      if (errors < 100)
        printf("%04X: mismatch %d != %d\n", i, d, ref);
    }
  }

  free(input_A.alloc);
  free(input_B.alloc);
  free(input_C.alloc);
  free(output.alloc);
  free(output_ref.alloc);

  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
    return -1;
  }
  return 0;
}
