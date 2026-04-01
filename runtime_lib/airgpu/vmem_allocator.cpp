// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "vmem_allocator.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err_ = (expr);                                                  \
    if (err_ != hipSuccess) {                                                  \
      fprintf(stderr, "airgpu: %s failed: %s (%d)\n", #expr,                  \
              hipGetErrorString(err_), static_cast<int>(err_));                 \
      abort();                                                                 \
    }                                                                          \
  } while (0)

size_t VMemAllocator::alignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

VMemAllocator::VMemAllocator(size_t heap_size) {
  // Allow override from environment
  if (const char *env = std::getenv("AIRGPU_HEAP_SIZE")) {
    heap_size = static_cast<size_t>(std::atol(env));
  }

  HIP_CHECK(hipGetDevice(&device_id_));
  HIP_CHECK(hipGetDeviceCount(&num_devices_));

  // Query allocation granularity
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device_id_;

  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(
      &gran, &prop, hipMemAllocationGranularityMinimum));
  granularity_ = gran;

  // Align heap size up to granularity
  heap_size_ = alignUp(heap_size, granularity_);

  // Reserve VA space
  hipDeviceptr_t va = 0;
  HIP_CHECK(hipMemAddressReserve(&va, heap_size_, granularity_, 0, 0));
  va_base_ = reinterpret_cast<void *>(va);

  // Build access descriptors for ALL GPUs (future-ready for symmetric heap)
  access_descs_.resize(num_devices_);
  for (int i = 0; i < num_devices_; i++) {
    access_descs_[i].location.type = hipMemLocationTypeDevice;
    access_descs_[i].location.id = i;
    access_descs_[i].flags = hipMemAccessFlagsProtReadWrite;
  }

  fprintf(stderr,
          "airgpu: VMemAllocator initialized on GPU %d "
          "(VA base=%p, heap=%zu MB, granularity=%zu KB, %d GPUs)\n",
          device_id_, va_base_, heap_size_ >> 20, granularity_ >> 10,
          num_devices_);
}

VMemAllocator::~VMemAllocator() {
  // Unmap and release all tracked allocations
  for (auto &rec : alloc_records_) {
    hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(rec.va_ptr), rec.size);
    hipMemRelease(rec.handle);
  }
  alloc_records_.clear();

  // Free reserved VA
  if (va_base_) {
    hipMemAddressFree(reinterpret_cast<hipDeviceptr_t>(va_base_), heap_size_);
    va_base_ = nullptr;
  }
}

void *VMemAllocator::allocate(size_t size_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (size_bytes == 0)
    size_bytes = 1;

  size_t aligned_size = alignUp(size_bytes, granularity_);
  size_t aligned_offset = alignUp(current_offset_, granularity_);

  if (aligned_offset + aligned_size > heap_size_) {
    fprintf(stderr,
            "airgpu: VMem heap exhausted "
            "(requested %zu, used %zu, total %zu)\n",
            aligned_size, aligned_offset, heap_size_);
    abort();
  }

  void *va_ptr = static_cast<char *>(va_base_) + aligned_offset;
  hipDeviceptr_t dptr = reinterpret_cast<hipDeviceptr_t>(va_ptr);

  // Create physical memory on this device
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device_id_;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, aligned_size, &prop, 0));

  // Map physical -> VA
  HIP_CHECK(hipMemMap(dptr, aligned_size, 0, handle, 0));

  // Set access for all GPUs
  HIP_CHECK(hipMemSetAccess(dptr, aligned_size, access_descs_.data(),
                            access_descs_.size()));

  // Track for cleanup
  alloc_records_.push_back({va_ptr, aligned_size, handle});
  current_offset_ = aligned_offset + aligned_size;

  return va_ptr;
}

void VMemAllocator::free(void *ptr) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it =
      std::find_if(alloc_records_.begin(), alloc_records_.end(),
                   [ptr](const AllocRecord &r) { return r.va_ptr == ptr; });
  if (it == alloc_records_.end()) {
    fprintf(stderr, "airgpu: free of unknown pointer %p\n", ptr);
    return;
  }

  hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(it->va_ptr), it->size);
  hipMemRelease(it->handle);
  alloc_records_.erase(it);
}
