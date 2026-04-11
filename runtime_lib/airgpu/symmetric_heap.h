// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Symmetric heap for multi-process multi-GPU memory sharing.
// All peer access is established at init time.

#pragma once

#include "vmem_allocator.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

// Tracks one imported peer mapping (for cleanup).
struct PeerMapping {
  hipMemGenericAllocationHandle_t handle;
  void *va;
  size_t size;
};

class SymmetricHeap {
public:
  // Collective constructor — all ranks must call with the same heap_size.
  // Reads RANK, WORLD_SIZE, LOCAL_RANK from environment.
  explicit SymmetricHeap(size_t heap_size);
  ~SymmetricHeap();

  // Allocate on the local symmetric heap.
  void *allocate(size_t size_bytes);

  // Free a symmetric heap allocation.
  void free(void *ptr);

  // Get the base VA for a specific rank's heap (as visible from this process).
  void *getHeapBase(int rank) const;

  // Get array of all heap bases (length = world_size).
  void **getHeapBases();

  int getRank() const { return rank_; }
  int getWorldSize() const { return world_size_; }
  int getDeviceId() const { return device_id_; }

  // Host-side barrier: all ranks synchronize via socket mesh.
  void barrier();

  SymmetricHeap(const SymmetricHeap &) = delete;
  SymmetricHeap &operator=(const SymmetricHeap &) = delete;

private:
  int rank_;
  int world_size_;
  int device_id_;

  VMemAllocator *allocator_;
  size_t heap_size_;
  size_t alloc_offset_ = 0; // bump pointer within pre-mapped heap

  // Socket mesh for fd passing (peer_rank -> socket fd)
  std::map<int, int> fd_mesh_;

  // heap_bases_[r] = VA base of rank r's heap as visible from THIS process
  std::vector<void *> heap_bases_;

  // Per-peer: imported mappings for cleanup
  std::map<int, std::vector<PeerMapping>> peer_mappings_;

  // Per-peer: separate VA reservation base
  std::map<int, void *> peer_va_bases_;

  // Set up peer access for all ranks.
  void establishPeerAccess();
};
