// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Drop-in replacement for LLVM's libmlir_rocm_runtime.so.
// Implements the same mgpu* C ABI but uses VMem-backed allocation
// (hipMemCreate/hipMemMap/hipMemSetAccess) instead of hipMalloc.
//
// Usage:
//   mlir-runner --entry-point-result=void \
//       --shared-libs=libairgpu.so \
//       final.mlir

#include "vmem_allocator.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_REPORT_IF_ERROR(expr)                                              \
    {                                                                          \
        hipError_t result = (expr);                                            \
        if (result != hipSuccess) {                                            \
            fprintf(stderr, "'%s' failed with '%s'\n", #expr,                  \
                    hipGetErrorString(result));                                 \
        }                                                                      \
    }

// Matches LLVM's StridedMemRefType used in mlir-runner
template <typename T, int N> struct StridedMemRefType {
    T *basePtr;
    T *data;
    int64_t offset;
    int64_t sizes[N];
    int64_t strides[N];
};

// ---------------------------------------------------------------------------
// Global allocator instance (constructor/destructor for library load/unload)
// ---------------------------------------------------------------------------

static VMemAllocator *g_allocator = nullptr;

__attribute__((constructor)) static void airgpu_runtime_init() {
    g_allocator = new VMemAllocator();
}

__attribute__((destructor)) static void airgpu_runtime_shutdown() {
    delete g_allocator;
    g_allocator = nullptr;
}

// ===========================================================================
// Module Management
// ===========================================================================

extern "C" hipModule_t mgpuModuleLoad(void *data, size_t /*gpuBlobSize*/) {
    hipModule_t module = nullptr;
    HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
    return module;
}

extern "C" hipModule_t mgpuModuleLoadJIT(void *data, int /*optLevel*/) {
    hipModule_t module = nullptr;
    HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
    return module;
}

extern "C" void mgpuModuleUnload(hipModule_t module) {
    HIP_REPORT_IF_ERROR(hipModuleUnload(module));
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                                const char *name) {
    hipFunction_t function = nullptr;
    HIP_REPORT_IF_ERROR(hipModuleGetFunction(&function, module, name));
    return function;
}

// ===========================================================================
// Kernel Launch
// ===========================================================================

extern "C" void mgpuLaunchKernel(hipFunction_t function, intptr_t gridX,
                                  intptr_t gridY, intptr_t gridZ,
                                  intptr_t blockX, intptr_t blockY,
                                  intptr_t blockZ, int32_t smem,
                                  hipStream_t stream, void **params,
                                  void **extra, size_t /*paramsCount*/) {
    HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(
        function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream,
        params, extra));
}

// ===========================================================================
// Stream Management
// ===========================================================================

extern "C" hipStream_t mgpuStreamCreate() {
    hipStream_t stream = nullptr;
    HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
    return stream;
}

extern "C" void mgpuStreamDestroy(hipStream_t stream) {
    HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(hipStream_t stream) {
    HIP_REPORT_IF_ERROR(hipStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(hipStream_t stream, hipEvent_t event) {
    HIP_REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, 0));
}

// ===========================================================================
// Event Management
// ===========================================================================

extern "C" hipEvent_t mgpuEventCreate() {
    hipEvent_t event = nullptr;
    HIP_REPORT_IF_ERROR(hipEventCreate(&event));
    return event;
}

extern "C" void mgpuEventDestroy(hipEvent_t event) {
    HIP_REPORT_IF_ERROR(hipEventDestroy(event));
}

extern "C" void mgpuEventSynchronize(hipEvent_t event) {
    HIP_REPORT_IF_ERROR(hipEventSynchronize(event));
}

extern "C" void mgpuEventRecord(hipEvent_t event, hipStream_t stream) {
    HIP_REPORT_IF_ERROR(hipEventRecord(event, stream));
}

// ===========================================================================
// Memory — VMem-backed (the key difference from LLVM's runtime)
// ===========================================================================

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, hipStream_t /*stream*/,
                               bool /*isHostShared*/) {
    return g_allocator->allocate(static_cast<size_t>(sizeBytes));
}

extern "C" void mgpuMemFree(void *ptr, hipStream_t /*stream*/) {
    if (ptr)
        g_allocator->free(ptr);
}

// ===========================================================================
// Memory Operations (standard HIP, same as LLVM)
// ===========================================================================

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes,
                            hipStream_t stream) {
    HIP_REPORT_IF_ERROR(
        hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
}

extern "C" void mgpuMemset32(void *dst, int value, size_t count,
                              hipStream_t stream) {
    HIP_REPORT_IF_ERROR(hipMemsetD32Async(
        reinterpret_cast<hipDeviceptr_t>(dst), value, count, stream));
}

extern "C" void mgpuMemset16(void *dst, short value, size_t count,
                              hipStream_t stream) {
    HIP_REPORT_IF_ERROR(hipMemsetD16Async(
        reinterpret_cast<hipDeviceptr_t>(dst), value, count, stream));
}

// ===========================================================================
// Host Memory Registration
// ===========================================================================

extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
    HIP_REPORT_IF_ERROR(
        hipHostRegister(ptr, sizeBytes, hipHostRegisterDefault));
}

extern "C" void mgpuMemHostRegisterMemRef(
    int64_t rank, StridedMemRefType<char, 1> *descriptor,
    int64_t elementSizeBytes) {
    if (rank > 0 && descriptor) {
        int64_t size = descriptor->sizes[0] * elementSizeBytes;
        HIP_REPORT_IF_ERROR(
            hipHostRegister(descriptor->data, size, hipHostRegisterDefault));
    }
}

extern "C" void mgpuMemHostUnregister(void *ptr) {
    HIP_REPORT_IF_ERROR(hipHostUnregister(ptr));
}

extern "C" void mgpuMemHostUnregisterMemRef(
    int64_t rank, StridedMemRefType<char, 1> *descriptor,
    int64_t elementSizeBytes) {
    if (descriptor) {
        HIP_REPORT_IF_ERROR(hipHostUnregister(descriptor->data));
    }
}

// ===========================================================================
// Device Management & MemRef Helpers
// ===========================================================================

extern "C" void mgpuSetDefaultDevice(int32_t device) {
    HIP_REPORT_IF_ERROR(hipSetDevice(device));
}

extern "C" StridedMemRefType<float, 1>
mgpuMemGetDeviceMemRef1dFloat(float *allocated, float *aligned, int64_t offset,
                               int64_t size, int64_t stride) {
    StridedMemRefType<float, 1> result;
    result.basePtr = allocated;
    result.data = aligned;
    result.offset = offset;
    result.sizes[0] = size;
    result.strides[0] = stride;
    return result;
}

extern "C" StridedMemRefType<int32_t, 1>
mgpuMemGetDeviceMemRef1dInt32(int32_t *allocated, int32_t *aligned,
                               int64_t offset, int64_t size, int64_t stride) {
    StridedMemRefType<int32_t, 1> result;
    result.basePtr = allocated;
    result.data = aligned;
    result.offset = offset;
    result.sizes[0] = size;
    result.strides[0] = stride;
    return result;
}
