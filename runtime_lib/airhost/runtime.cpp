// SPDX-License-Identifier: MIT
// Copyright (C) 2023, Advanced Micro Devices, Inc.

#include "runtime.h"

#include "debug.h"

namespace air {
namespace rocm {

Runtime *Runtime::runtime_ = nullptr;

void Runtime::Init() {
  if (!runtime_) {
    runtime_ = new Runtime();
  }

  runtime_->FindAieAgents();
  runtime_->InitMemSegments();
}

void Runtime::ShutDown() { delete runtime_; }

void *Runtime::AllocateMemory(size_t size) {
  void *mem(nullptr);

  hsa_amd_memory_pool_allocate(global_mem_pool_, size, 0, &mem);

  return mem;
}

void Runtime::FreeMemory(void *ptr) { hsa_amd_memory_pool_free(ptr); }

hsa_status_t Runtime::IterateAgents(hsa_agent_t agent, void *data) {
  hsa_status_t status(HSA_STATUS_SUCCESS);
  hsa_device_type_t device_type;
  std::vector<hsa_agent_t> *aie_agents(nullptr);

  if (!data) {
    status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
    return status;
  }

  aie_agents = static_cast<std::vector<hsa_agent_t> *>(data);
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_AIE) {
    aie_agents->push_back(agent);
  }

  return status;
}

hsa_status_t Runtime::IterateMemPool(hsa_amd_memory_pool_t pool, void *data) {
  hsa_status_t status(HSA_STATUS_SUCCESS);
  hsa_region_segment_t segment_type;
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                        &segment_type);
  if (segment_type == HSA_REGION_SEGMENT_GLOBAL) {
    debug_print("Runtime: found global segment");
    *reinterpret_cast<hsa_amd_memory_pool_t *>(data) = pool;
  }

  return status;
}

void Runtime::FindAieAgents() {
  hsa_iterate_agents(&Runtime::IterateAgents,
                     reinterpret_cast<void *>(&aie_agents_));
  debug_print("Runtime: found ", aie_agents_.size(), " AIE agents");
}

void Runtime::InitMemSegments() {
  debug_print("Runtime: initializing memory pools");
  hsa_amd_agent_iterate_memory_pools(
      aie_agents_.front(), &Runtime::IterateMemPool,
      reinterpret_cast<void *>(&global_mem_pool_));
}

} // namespace rocm
} // namespace air
