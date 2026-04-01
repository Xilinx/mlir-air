// SPDX-License-Identifier: MIT
// Smoke test: verify VMem allocator works by allocating, writing, reading back.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

// Import mgpu* functions from libairgpu.so
extern "C" {
void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t stream, bool isHostShared);
void mgpuMemFree(void *ptr, hipStream_t stream);
void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, hipStream_t stream);
hipStream_t mgpuStreamCreate();
void mgpuStreamDestroy(hipStream_t stream);
void mgpuStreamSynchronize(hipStream_t stream);
void mgpuSetDefaultDevice(int32_t device);
}

int main() {
  printf("=== VMem Allocator Smoke Test ===\n");

  mgpuSetDefaultDevice(0);

  const size_t N = 1024;
  const size_t size = N * sizeof(float);

  // Allocate device memory via VMem
  printf("Allocating %zu bytes via VMem...\n", size);
  void *d_buf = mgpuMemAlloc(size, nullptr, false);
  if (!d_buf) {
    fprintf(stderr, "mgpuMemAlloc returned null\n");
    return 1;
  }
  printf("  Device pointer: %p\n", d_buf);

  // Prepare host data
  float *h_input = (float *)malloc(size);
  float *h_output = (float *)malloc(size);
  for (size_t i = 0; i < N; i++) {
    h_input[i] = static_cast<float>(i);
  }
  memset(h_output, 0, size);

  // Copy host -> device -> host
  hipStream_t stream = mgpuStreamCreate();
  printf("Copying host -> device...\n");
  mgpuMemcpy(d_buf, h_input, size, stream);
  printf("Copying device -> host...\n");
  mgpuMemcpy(h_output, d_buf, size, stream);
  mgpuStreamSynchronize(stream);

  // Verify
  int mismatches = 0;
  for (size_t i = 0; i < N; i++) {
    if (h_output[i] != h_input[i]) {
      fprintf(stderr, "Mismatch at [%zu]: expected %f, got %f\n", i,
              h_input[i], h_output[i]);
      mismatches++;
      if (mismatches > 5)
        break;
    }
  }

  if (mismatches == 0) {
    printf("All %zu elements verified correctly.\n", N);
  } else {
    printf("%d mismatches found!\n", mismatches);
  }

  // Cleanup
  mgpuMemFree(d_buf, stream);
  mgpuStreamDestroy(stream);
  free(h_input);
  free(h_output);

  printf("=== %s ===\n", mismatches == 0 ? "PASSED" : "FAILED");
  return mismatches > 0 ? 1 : 0;
}
