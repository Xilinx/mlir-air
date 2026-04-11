// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Multi-process test for the symmetric heap runtime.
// Launch with run_symmetric_heap_test.sh or set RANK/WORLD_SIZE/LOCAL_RANK
// and run directly.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>

// Forward-declare the mgpu* C ABI
extern "C" {
void mgpuSymmetricHeapInit(uint64_t heap_size);
void mgpuSymmetricHeapDestroy();
int32_t mgpuGetRank();
int32_t mgpuGetWorldSize();
void *mgpuSymmetricAlloc(uint64_t sizeBytes, hipStream_t stream);
void mgpuSymmetricFree(void *ptr, hipStream_t stream);
void *mgpuGetHeapBase(int32_t rank);
void **mgpuGetHeapBases();
void mgpuBarrier();
void mgpuSetDevice(int32_t device_id);
}

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error: %s (%d) at %s:%d\n", hipGetErrorString(err), \
              err, __FILE__, __LINE__);                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  // Line-buffer stdout so output appears immediately when piped through sed
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setvbuf(stderr, nullptr, _IOLBF, 0);

  // ---- Test 1: Init ----
  printf("[test] Initializing symmetric heap (256 MB)...\n");
  mgpuSymmetricHeapInit(256ULL << 20);

  int rank = mgpuGetRank();
  int world_size = mgpuGetWorldSize();
  printf("[test] rank=%d world_size=%d\n", rank, world_size);
  assert(rank >= 0 && rank < world_size);

  // ---- Test 2: Allocate ----
  size_t N = 1024;
  float *buf =
      static_cast<float *>(mgpuSymmetricAlloc(N * sizeof(float), nullptr));
  assert(buf != nullptr);
  printf("[test] rank %d: allocated %zu floats at %p\n", rank, N, buf);

  // ---- Test 3: Heap bases ----
  void **bases = mgpuGetHeapBases();
  assert(bases != nullptr);
  for (int r = 0; r < world_size; r++) {
    printf("[test] rank %d: heap_bases[%d] = %p\n", rank, r, bases[r]);
    assert(bases[r] != nullptr);
  }

  // ---- Test 4: Write local pattern and barrier ----
  // Each rank writes its rank value to every element
  float rank_f = static_cast<float>(rank + 1); // +1 to avoid 0.0
  HIP_CHECK(hipMemset(buf, 0, N * sizeof(float)));
  // Use a simple kernel-free approach: write from host
  std::vector<float> host_buf(N, rank_f);
  HIP_CHECK(hipMemcpy(buf, host_buf.data(), N * sizeof(float),
                      hipMemcpyHostToDevice));

  mgpuBarrier();

  // ---- Test 5: Read from peer's heap ----
  if (world_size > 1) {
    int peer = (rank + 1) % world_size;
    float *peer_base = static_cast<float *>(bases[peer]);

    // The peer wrote (peer+1) to its buffer. The buffer is at offset 0
    // from the peer's heap base, so we can read it directly.
    // But we need to know the offset of `buf` within the heap.
    // Since buf is the first allocation, it's at offset =
    // granularity-aligned(0). Let's compute the offset from our local base.
    uintptr_t local_offset = reinterpret_cast<uintptr_t>(buf) -
                             reinterpret_cast<uintptr_t>(bases[rank]);

    float *peer_buf = reinterpret_cast<float *>(
        reinterpret_cast<uintptr_t>(peer_base) + local_offset);

    // Copy peer data to a local buffer first (D2D), then read to host.
    // Direct D2H from imported VA may not be supported.
    float *local_copy = nullptr;
    HIP_CHECK(hipMalloc(&local_copy, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(local_copy, peer_buf, N * sizeof(float),
                        hipMemcpyDeviceToDevice));

    std::vector<float> readback(N);
    HIP_CHECK(hipMemcpy(readback.data(), local_copy, N * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(local_copy));

    float expected = static_cast<float>(peer + 1);
    int mismatches = 0;
    for (size_t i = 0; i < N; i++) {
      if (readback[i] != expected) {
        if (mismatches < 5)
          fprintf(
              stderr,
              "[test] rank %d: MISMATCH at [%zu]: got %.1f, expected %.1f\n",
              rank, i, readback[i], expected);
        mismatches++;
      }
    }

    if (mismatches == 0) {
      printf("[test] rank %d: cross-rank read from rank %d PASSED "
             "(all %zu values = %.1f)\n",
             rank, peer, N, expected);
    } else {
      fprintf(stderr,
              "[test] rank %d: cross-rank read FAILED (%d/%zu "
              "mismatches)\n",
              rank, mismatches, N);
      return 1;
    }
  }

  mgpuBarrier();

  // ---- Cleanup ----
  mgpuSymmetricFree(buf, nullptr);
  mgpuSymmetricHeapDestroy();

  printf("[test] rank %d/%d: ALL PASSED\n", rank, world_size);
  return 0;
}
