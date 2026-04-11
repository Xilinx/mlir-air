// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "symmetric_heap.h"
#include "fd_passing.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err_ = (expr);                                                  \
    if (err_ != hipSuccess) {                                                  \
      fprintf(stderr, "airgpu: %s failed: %s (%d)\n", #expr,                   \
              hipGetErrorString(err_), static_cast<int>(err_));                \
      abort();                                                                 \
    }                                                                          \
  } while (0)

static int envInt(const char *name, int fallback) {
  const char *val = std::getenv(name);
  if (!val)
    return fallback;
  return atoi(val);
}

SymmetricHeap::SymmetricHeap(size_t heap_size) {
  // Read rank info from environment
  rank_ = envInt("RANK", -1);
  world_size_ = envInt("WORLD_SIZE", -1);
  device_id_ = envInt("LOCAL_RANK", rank_);

  if (rank_ < 0 || world_size_ < 1) {
    fprintf(stderr,
            "airgpu: RANK and WORLD_SIZE environment variables required\n");
    abort();
  }

  fprintf(stderr, "airgpu: rank %d/%d initializing symmetric heap (%zu MB)\n",
          rank_, world_size_, heap_size >> 20);

  // Set GPU for this rank
  HIP_CHECK(hipSetDevice(device_id_));

  heap_size_ = heap_size;

  // Create local VMem allocator and pre-map the full heap.
  // This maps the entire heap as one physical chunk so peers can import it.
  allocator_ = new VMemAllocator(heap_size);
  allocator_->allocate(heap_size);

  // Initialize heap_bases — local rank gets its own VA base
  heap_bases_.resize(world_size_, nullptr);
  heap_bases_[rank_] = allocator_->getVaBase();

  if (world_size_ > 1) {
    // Set up socket mesh for fd passing
    fd_mesh_ = setupFdMesh(rank_, world_size_);

    // Establish peer access (exchange fds, import, map)
    establishPeerAccess();
  }

  // Print heap bases for debugging
  for (int r = 0; r < world_size_; r++) {
    fprintf(stderr, "airgpu: rank %d: heap_bases[%d] = %p\n", rank_, r,
            heap_bases_[r]);
  }
}

SymmetricHeap::~SymmetricHeap() {
  // Synchronize before cleanup
  hipError_t sync_err = hipDeviceSynchronize();
  if (sync_err != hipSuccess)
    fprintf(stderr, "airgpu: hipDeviceSynchronize failed: %s\n",
            hipGetErrorString(sync_err));

  // Unmap and release all peer imports
  for (auto &[peer, mappings] : peer_mappings_) {
    for (auto &m : mappings) {
      hipError_t err =
          hipMemUnmap(reinterpret_cast<hipDeviceptr_t>(m.va), m.size);
      if (err != hipSuccess)
        fprintf(stderr, "airgpu: hipMemUnmap(%p) failed: %s\n", m.va,
                hipGetErrorString(err));
      err = hipMemRelease(m.handle);
      if (err != hipSuccess)
        fprintf(stderr, "airgpu: hipMemRelease failed: %s\n",
                hipGetErrorString(err));
    }
  }

  // Free peer VA reservations
  for (auto &[peer, va] : peer_va_bases_) {
    hipError_t err = hipMemAddressFree(reinterpret_cast<hipDeviceptr_t>(va),
                                       allocator_->getHeapSize());
    if (err != hipSuccess)
      fprintf(stderr, "airgpu: hipMemAddressFree(%p) failed: %s\n", va,
              hipGetErrorString(err));
  }

  // Teardown socket mesh
  teardownFdMesh(fd_mesh_);

  // Delete local allocator
  delete allocator_;
  allocator_ = nullptr;

  fprintf(stderr, "airgpu: rank %d symmetric heap destroyed\n", rank_);
}

void *SymmetricHeap::allocate(size_t size_bytes) {
  // Simple bump allocator within the pre-mapped heap.
  // No new hipMemCreate/hipMemMap — the full heap is already mapped.
  size_t granularity = allocator_->getGranularity();
  size_t aligned = (size_bytes + granularity - 1) & ~(granularity - 1);
  size_t offset = (alloc_offset_ + granularity - 1) & ~(granularity - 1);

  if (offset + aligned > heap_size_) {
    fprintf(stderr,
            "airgpu: symmetric heap out of memory "
            "(requested %zu, used %zu, total %zu)\n",
            size_bytes, offset, heap_size_);
    abort();
  }

  void *ptr = static_cast<char *>(allocator_->getVaBase()) + offset;
  alloc_offset_ = offset + aligned;
  return ptr;
}

void SymmetricHeap::free(void * /*ptr*/) {
  // Bump allocator — individual frees are a no-op.
  // The full heap is released when SymmetricHeap is destroyed.
}

void *SymmetricHeap::getHeapBase(int rank) const {
  if (rank < 0 || rank >= world_size_)
    return nullptr;
  return heap_bases_[rank];
}

void **SymmetricHeap::getHeapBases() { return heap_bases_.data(); }

void SymmetricHeap::barrier() {
  if (world_size_ <= 1)
    return;

  // Simple O(N) barrier over socket mesh.
  // Protocol: for each pair, the lower rank sends first and the higher rank
  // receives first. This avoids deadlock.
  char token = 0;
  for (auto &[peer, sock] : fd_mesh_) {
    if (rank_ < peer) {
      if (sendAll(sock, &token, 1) < 0 || recvAll(sock, &token, 1) < 0) {
        fprintf(stderr, "airgpu: barrier failed (rank=%d, peer=%d)\n", rank_,
                peer);
        abort();
      }
    } else {
      if (recvAll(sock, &token, 1) < 0 || sendAll(sock, &token, 1) < 0) {
        fprintf(stderr, "airgpu: barrier failed (rank=%d, peer=%d)\n", rank_,
                peer);
        abort();
      }
    }
  }
}

void SymmetricHeap::establishPeerAccess() {
  // Step 1: Exchange base addresses via sockets
  // Higher rank sends first to avoid deadlock
  uintptr_t my_base = reinterpret_cast<uintptr_t>(allocator_->getVaBase());

  for (auto &[peer, sock] : fd_mesh_) {
    uintptr_t peer_base = 0;
    if (peer < rank_) {
      // Send first, then recv
      sendAll(sock, &my_base, sizeof(my_base));
      recvAll(sock, &peer_base, sizeof(peer_base));
    } else {
      // Recv first, then send
      recvAll(sock, &peer_base, sizeof(peer_base));
      sendAll(sock, &my_base, sizeof(my_base));
    }
    // Store the peer's original base for reference (not the mapped VA yet)
    (void)peer_base;
  }

  barrier();

  // Step 2: Export local allocation handles and exchange with peers
  const auto &records = allocator_->getAllocRecords();
  size_t heap_size = allocator_->getHeapSize();
  size_t granularity = allocator_->getGranularity();

  // Build local access descriptor for this device
  hipMemAccessDesc access_desc = {};
  access_desc.location.type = hipMemLocationTypeDevice;
  access_desc.location.id = device_id_;
  access_desc.flags = hipMemAccessFlagsProtReadWrite;

  for (auto &[peer, sock] : fd_mesh_) {
    // Reserve a SEPARATE VA range for this peer's heap
    // (critical for gfx950 — imported handles can't share VA with local ones)
    hipDeviceptr_t peer_va = 0;
    HIP_CHECK(hipMemAddressReserve(&peer_va, heap_size, granularity, 0, 0));
    peer_va_bases_[peer] = reinterpret_cast<void *>(peer_va);

    // Exchange allocation count
    uint32_t my_count = static_cast<uint32_t>(records.size());
    uint32_t peer_count = 0;

    if (peer < rank_) {
      sendAll(sock, &my_count, sizeof(my_count));
      recvAll(sock, &peer_count, sizeof(peer_count));
    } else {
      recvAll(sock, &peer_count, sizeof(peer_count));
      sendAll(sock, &my_count, sizeof(my_count));
    }

    // Exchange each allocation's metadata and fd
    // Send: (offset, size, fd) for each local record
    // Recv: (offset, size, fd) for each peer record
    for (uint32_t i = 0; i < std::max(my_count, peer_count); i++) {
      if (peer < rank_) {
        // Send local record, then recv peer record
        if (i < my_count) {
          uint64_t offset =
              reinterpret_cast<uintptr_t>(records[i].va_ptr) -
              reinterpret_cast<uintptr_t>(allocator_->getVaBase());
          uint64_t size = records[i].size;
          sendAll(sock, &offset, sizeof(offset));
          sendAll(sock, &size, sizeof(size));

          int fd = -1;
          HIP_CHECK(hipMemExportToShareableHandle(
              &fd, records[i].handle, hipMemHandleTypePosixFileDescriptor, 0));
          sendFd(sock, fd);
          close(fd);
        }
        if (i < peer_count) {
          uint64_t offset = 0, size = 0;
          recvAll(sock, &offset, sizeof(offset));
          recvAll(sock, &size, sizeof(size));
          int peer_fd = recvFd(sock);

          // Import the peer's handle
          hipMemGenericAllocationHandle_t imported = {};
          HIP_CHECK(hipMemImportFromShareableHandle(
              &imported,
              reinterpret_cast<void *>(static_cast<intptr_t>(peer_fd)),
              hipMemHandleTypePosixFileDescriptor));
          close(peer_fd);

          // Map into the peer's VA reservation
          hipDeviceptr_t target = static_cast<hipDeviceptr_t>(
              static_cast<char *>(peer_va) + offset);
          HIP_CHECK(hipMemMap(target, size, 0, imported, 0));
          HIP_CHECK(hipMemSetAccess(target, size, &access_desc, 1));

          peer_mappings_[peer].push_back(
              {imported, reinterpret_cast<void *>(target), size});
        }
      } else {
        // Recv peer record first, then send local
        if (i < peer_count) {
          uint64_t offset = 0, size = 0;
          recvAll(sock, &offset, sizeof(offset));
          recvAll(sock, &size, sizeof(size));
          int peer_fd = recvFd(sock);

          hipMemGenericAllocationHandle_t imported = {};
          HIP_CHECK(hipMemImportFromShareableHandle(
              &imported,
              reinterpret_cast<void *>(static_cast<intptr_t>(peer_fd)),
              hipMemHandleTypePosixFileDescriptor));
          close(peer_fd);

          hipDeviceptr_t target = static_cast<hipDeviceptr_t>(
              static_cast<char *>(peer_va) + offset);
          HIP_CHECK(hipMemMap(target, size, 0, imported, 0));
          HIP_CHECK(hipMemSetAccess(target, size, &access_desc, 1));

          peer_mappings_[peer].push_back(
              {imported, reinterpret_cast<void *>(target), size});
        }
        if (i < my_count) {
          uint64_t offset =
              reinterpret_cast<uintptr_t>(records[i].va_ptr) -
              reinterpret_cast<uintptr_t>(allocator_->getVaBase());
          uint64_t size = records[i].size;
          sendAll(sock, &offset, sizeof(offset));
          sendAll(sock, &size, sizeof(size));

          int fd = -1;
          HIP_CHECK(hipMemExportToShareableHandle(
              &fd, records[i].handle, hipMemHandleTypePosixFileDescriptor, 0));
          sendFd(sock, fd);
          close(fd);
        }
      }
    }

    heap_bases_[peer] = reinterpret_cast<void *>(peer_va);
  }

  barrier();
  fprintf(stderr, "airgpu: rank %d peer access established\n", rank_);
}
