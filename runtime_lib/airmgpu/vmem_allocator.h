// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// VMem-backed GPU memory allocator for MLIR-AIR.
// Uses HIP VMem APIs (hipMemCreate, hipMemMap, hipMemSetAccess) instead of
// hipMalloc. This makes allocated memory future-ready for symmetric heap /
// multi-GPU access via XGMI without reallocation.

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

struct AllocRecord {
    void *va_ptr;
    size_t size;
    hipMemGenericAllocationHandle_t handle;
};

class VMemAllocator {
public:
    // heap_size: total VA space to reserve (default 1GB, env AIR_MGPU_HEAP_SIZE)
    explicit VMemAllocator(size_t heap_size = 1ULL << 30);
    ~VMemAllocator();

    // Allocate device memory backed by VMem.
    // Returns device pointer accessible from all GPUs (access granted at alloc).
    void *allocate(size_t size_bytes);

    // Free a previously allocated pointer.
    void free(void *ptr);

    // Accessors for future symmetric heap extension
    void *get_va_base() const { return va_base_; }
    size_t get_heap_size() const { return heap_size_; }
    size_t get_granularity() const { return granularity_; }
    int get_device_id() const { return device_id_; }
    int get_num_devices() const { return num_devices_; }

    VMemAllocator(const VMemAllocator &) = delete;
    VMemAllocator &operator=(const VMemAllocator &) = delete;

private:
    void *va_base_ = nullptr;
    size_t heap_size_;
    size_t current_offset_ = 0;
    size_t granularity_;
    int device_id_ = 0;
    int num_devices_ = 1;

    std::vector<hipMemAccessDesc> access_descs_;
    std::vector<AllocRecord> alloc_records_;
    std::mutex mutex_;

    static size_t align_up(size_t value, size_t alignment);
};
