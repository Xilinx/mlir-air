//===- host.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air.hpp"
#include "air_host.h"
#include "air_host_impl.h"
#include "runtime.h"
#include "test_library.h"

#include <assert.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream> // ifstream
#include <iomanip> // setbase()
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_NUM_ROWS 9
#define XAIE_NUM_COLS 50
#define XAIE_COL_SHIFT 23
#define XAIE_ROW_SHIFT 18
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 0
#define XAIE_RES_TILE_NUM_ROWS 0
#define XAIE_AIE_TILE_ROW_START 1
#define XAIE_AIE_TILE_NUM_ROWS 8

#define SYSFS_PATH_MAX 63

#define BOUNCE_BUFFER_SIZE 0x8000

// temporary solution to stash some state
extern "C" {

air_rt_herd_desc_t _air_host_active_herd = {nullptr, nullptr, nullptr};
air_rt_segment_desc_t _air_host_active_segment = {nullptr, nullptr, nullptr};
aie_libxaie_ctx_t *_air_host_active_libxaie = nullptr;
uint32_t *_air_host_bram_ptr = nullptr;
uint64_t _air_host_bram_paddr = 0;
air_module_handle_t _air_host_active_module = (air_module_handle_t) nullptr;

const char vck5000_driver_name[] = "/dev/amdair";
}

// Determining if an hsa agent is an AIE agent or not
hsa_status_t find_aie(hsa_agent_t agent, void *data) {
  hsa_status_t status(HSA_STATUS_SUCCESS);
  hsa_device_type_t device_type;
  std::vector<hsa_agent_t> *aie_agents = nullptr;

  if (!data) {
    status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
    printf("find_aie: INVALID ARGUMENT\n");
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

hsa_status_t air_init() {
  printf("%s\n", __func__);

  hsa_status_t hsa_ret = hsa_init();
  air::rocm::Runtime::Init();

  if (hsa_ret != HSA_STATUS_SUCCESS) {
    std::cerr << "hsa_init failed" << std::endl;
    return hsa_ret;
  }

  if (_air_host_active_libxaie == nullptr)
    air_init_libxaie();

  if (_air_host_active_libxaie == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t air_shut_down() {
  if (!_air_host_active_libxaie)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;

  if (_air_host_active_module)
    air_module_unload(_air_host_active_module);

  if (_air_host_active_libxaie)
    air_deinit_libxaie((air_libxaie_ctx_t)_air_host_active_libxaie);

  hsa_status_t hsa_ret = hsa_shut_down();
  if (hsa_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] hsa_shut_down() failed\n");
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
}

air_libxaie_ctx_t air_get_libxaie_ctx() {
  return (air_libxaie_ctx_t)_air_host_active_libxaie;
}

air_libxaie_ctx_t air_init_libxaie(uint32_t device_id) {
  if (_air_host_active_libxaie)
    return (air_libxaie_ctx_t)_air_host_active_libxaie;

  aie_libxaie_ctx_t *xaie =
      (aie_libxaie_ctx_t *)malloc(sizeof(aie_libxaie_ctx_t));
  if (!xaie)
    return (air_libxaie_ctx_t) nullptr;

  xaie->AieConfigPtr.AieGen = XAIE_DEV_GEN_AIE;
  xaie->AieConfigPtr.ColShift = XAIE_COL_SHIFT;
  xaie->AieConfigPtr.RowShift = XAIE_ROW_SHIFT;
  xaie->AieConfigPtr.NumRows = XAIE_NUM_ROWS;
  xaie->AieConfigPtr.NumCols = XAIE_NUM_COLS;
  xaie->AieConfigPtr.ShimRowNum = XAIE_SHIM_ROW;
  xaie->AieConfigPtr.MemTileRowStart = XAIE_RES_TILE_ROW_START;
  xaie->AieConfigPtr.MemTileNumRows = XAIE_RES_TILE_NUM_ROWS;
  xaie->AieConfigPtr.AieTileRowStart = XAIE_AIE_TILE_ROW_START;
  xaie->AieConfigPtr.AieTileNumRows = XAIE_AIE_TILE_NUM_ROWS;
  xaie->AieConfigPtr.PartProp = {0};
  xaie->DevInst = {0};

  char sysfs_path[SYSFS_PATH_MAX + 1];
  if (snprintf(sysfs_path, SYSFS_PATH_MAX, "/sys/class/amdair/amdair/%02u",
               device_id) == SYSFS_PATH_MAX)
    sysfs_path[SYSFS_PATH_MAX] = 0;

  XAie_BackendType backend;
  xaie->AieConfigPtr.Backend = XAIE_IO_BACKEND_AMDAIR;
  backend = XAIE_IO_BACKEND_AMDAIR;
  xaie->AieConfigPtr.BaseAddr = 0;
  xaie->DevInst.IOInst = (void *)sysfs_path;

  if (XAie_CfgInitialize(&(xaie->DevInst), &(xaie->AieConfigPtr)) != XAIE_OK) {
    printf("[ERROR] Failed to configure libxaie\n");
    return (air_libxaie_ctx_t) nullptr;
  }

  XAie_PmRequestTiles(&(xaie->DevInst), NULL, 0);

  _air_host_active_libxaie = xaie;
  return (air_libxaie_ctx_t)xaie;
}

void air_deinit_libxaie(air_libxaie_ctx_t _xaie) {
  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)_xaie;
  if (xaie == _air_host_active_libxaie) {
    XAie_Finish(&(xaie->DevInst));

    if (xaie->AieConfigPtr.BaseAddr)
      munmap((void *)xaie->AieConfigPtr.BaseAddr, 0x20000000);

    _air_host_active_libxaie = nullptr;
  }
  free(xaie);
}

air_module_handle_t air_module_load_from_file(const char *filename,
                                              hsa_agent_t *agent,
                                              hsa_queue_t *q,
                                              uint32_t device_id) {
  if (_air_host_active_module)
    air_module_unload(_air_host_active_module);

  air_module_handle_t handle;
  void *_handle = dlopen(filename, RTLD_NOW);
  if (!_handle) {
    printf("%s\n", dlerror());
    return 0;
  }
  _air_host_active_module = (air_module_handle_t)_handle;
  _air_host_active_herd = {q, agent, nullptr};
  _air_host_active_segment = {q, agent, nullptr};

  _air_host_bram_ptr = (uint32_t *)air_malloc(BOUNCE_BUFFER_SIZE);

  assert((_air_host_bram_ptr != MAP_FAILED) &&
         "Failed to map scratch bram location");

  return (air_module_handle_t)_handle;
}

int32_t air_module_unload(air_module_handle_t handle) {
  if (!handle)
    return -1;

  if (auto module_desc = air_module_get_desc(handle)) {
    for (int i = 0; i < module_desc->segment_length; i++) {
      for (int j = 0; j < module_desc->segment_descs[i]->herd_length; j++) {
        auto herd_desc = module_desc->segment_descs[i]->herd_descs[j];
        if (herd_desc == _air_host_active_herd.herd_desc) {
          if (_air_host_active_segment.q) {
            hsa_queue_destroy(_air_host_active_segment.q);
          }
          _air_host_active_herd = {nullptr, nullptr};
          _air_host_active_segment = {nullptr, nullptr, nullptr};
        }
      }
    }
  }
  if (_air_host_active_module == handle) {
    _air_host_active_module = (air_module_handle_t) nullptr;

    air_free(_air_host_bram_ptr);
    _air_host_bram_paddr = 0;
  }

  return dlclose((void *)handle);
}

air_herd_desc_t *air_herd_get_desc(air_module_handle_t handle,
                                   air_segment_desc_t *segment_desc,
                                   const char *herd_name) {
  if (!handle)
    return nullptr;
  if (!segment_desc)
    return nullptr;

  auto module_desc = air_module_get_desc(handle);
  if (!module_desc)
    return nullptr;

  if (!air_segment_get_desc(handle, segment_desc->name))
    return nullptr;

  for (int i = 0; i < segment_desc->herd_length; i++) {
    auto herd_desc = segment_desc->herd_descs[i];
    if (!strncmp(herd_name, herd_desc->name, herd_desc->name_length))
      return herd_desc;
  }
  return nullptr;
}

air_segment_desc_t *air_segment_get_desc(air_module_handle_t handle,
                                         const char *segment_name) {
  if (!handle)
    return nullptr;

  auto module_desc = air_module_get_desc(handle);
  if (!module_desc)
    return nullptr;

  for (int i = 0; i < module_desc->segment_length; i++) {
    auto segment_desc = module_desc->segment_descs[i];
    if (!strncmp(segment_name, segment_desc->name, segment_desc->name_length)) {
      return segment_desc;
    }
  }
  return nullptr;
}

air_module_desc_t *air_module_get_desc(air_module_handle_t handle) {
  if (!handle)
    return nullptr;
  return (air_module_desc_t *)dlsym((void *)handle,
                                    "__airrt_module_descriptor");
}

uint64_t air_segment_load(const char *name) {

  assert(_air_host_active_libxaie);

  auto segment_desc = air_segment_get_desc(_air_host_active_module, name);
  if (!segment_desc) {
    printf("Failed to locate segment descriptor '%s'!\n", name);
    assert(0);
  }

  XAie_Finish(&(_air_host_active_libxaie->DevInst));

  // Setting the driver libxaie backend back up
  // Currently only targetting device 0
  _air_host_active_libxaie->AieConfigPtr.Backend = XAIE_IO_BACKEND_AMDAIR;
  _air_host_active_libxaie->DevInst.IOInst =
      (void *)"/sys/class/amdair/amdair/00";

  XAie_CfgInitialize(&(_air_host_active_libxaie->DevInst),
                     &(_air_host_active_libxaie->AieConfigPtr));
  XAie_PmRequestTiles(&(_air_host_active_libxaie->DevInst), NULL, 0);

  //
  // Set up a 1x3 herd starting 7,0
  //
  uint64_t wr_idx =
      hsa_queue_add_write_index_relaxed(_air_host_active_segment.q, 1);
  uint64_t packet_id = wr_idx % _air_host_active_segment.q->size;
  hsa_agent_dispatch_packet_t shim_pkt;
  air_packet_device_init(&shim_pkt, XAIE_NUM_COLS);
  air_queue_dispatch_and_wait(_air_host_active_segment.agent,
                              _air_host_active_segment.q, packet_id, wr_idx,
                              &shim_pkt);

  wr_idx = hsa_queue_add_write_index_relaxed(_air_host_active_segment.q, 1);
  packet_id = wr_idx % _air_host_active_segment.q->size;
  hsa_agent_dispatch_packet_t segment_pkt;
  air_packet_segment_init(&segment_pkt, 0, 0, 50, 1, 8);
  air_queue_dispatch_and_wait(_air_host_active_segment.agent,
                              _air_host_active_segment.q, packet_id, wr_idx,
                              &segment_pkt);

  std::string segment_name(segment_desc->name, segment_desc->name_length);

  std::string func_name = "__airrt_" + segment_name + "_aie_functions";
  air_rt_aie_functions_t *mlir = (air_rt_aie_functions_t *)dlsym(
      (void *)_air_host_active_module, func_name.c_str());

  if (mlir) {
    // printf("configuring segment: '%s'\n", segment_name.c_str());
    assert(mlir->configure_cores);
    assert(mlir->configure_switchboxes);
    assert(mlir->initialize_locks);
    assert(mlir->configure_dmas);
    assert(mlir->start_cores);
    mlir->configure_cores(_air_host_active_libxaie);
    mlir->configure_switchboxes(_air_host_active_libxaie);
    mlir->initialize_locks(_air_host_active_libxaie);
    mlir->configure_dmas(_air_host_active_libxaie);
    mlir->start_cores(_air_host_active_libxaie);
  } else {
    printf("Failed to locate segment '%s' configuration functions!\n",
           segment_name.c_str());
    assert(0);
  }
  _air_host_active_segment.segment_desc = segment_desc;
  return 0;
}

uint64_t air_herd_load(const char *name) {

  // If no segment is loaded, load the segment associated with this herd
  if (!_air_host_active_segment.segment_desc) {
    bool loaded = false;
    if (auto module_desc = air_module_get_desc(_air_host_active_module)) {
      for (int i = 0; !loaded && i < module_desc->segment_length; i++) {
        for (int j = 0;
             !loaded && j < module_desc->segment_descs[i]->herd_length; j++) {
          auto herd_desc = module_desc->segment_descs[i]->herd_descs[j];
          // use the segment of the first herd with a matching name
          if (!strncmp(name, herd_desc->name, herd_desc->name_length)) {
            air_segment_load(module_desc->segment_descs[i]->name);
            loaded = true; // break
          }
        }
      }
    }
  }
  auto herd_desc = air_herd_get_desc(
      _air_host_active_module, _air_host_active_segment.segment_desc, name);
  // In some scenarios load_segment is not called. This is a temporary hack
  // to support that case.
  if (!herd_desc) {
    if (_air_host_active_segment.segment_desc) {
      _air_host_active_segment.segment_desc = 0;
      return air_herd_load(name);
    }
    printf("Failed to locate herd descriptor '%s'!\n", name);
    assert(0);
  }
  _air_host_active_herd.herd_desc = herd_desc;

  return 0;
}

uint64_t air_wait_all(std::vector<uint64_t> &signals) {
  hsa_queue_t *q = _air_host_active_segment.q;
  if (!q) {
    printf("WARNING: no queue provided, air_wait_all will return without "
           "waiting\n");
    return 0;
  }

  // Containing all of the barrier packets. This is needed because
  // we might have to wait on more than 5 signals.
  std::vector<hsa_barrier_and_packet_t> packets;

  // iterate over the signals in chunks of 5
  while (signals.size()) {
    if (signals.size() < 5)
      signals.resize(5, 0);

    // Vector which contains the handles of the signals that we are going to
    // wait on
    std::vector<hsa_signal_t> signals_in_pkt;
    bool non_zero = false;
    for (auto s : signals) {
      if (s) {
        // Push back the proper signal
        signals_in_pkt.push_back(*reinterpret_cast<hsa_signal_t *>(s));
        non_zero = true;
      } else {
        // Create a dummy signal that will have a handle of 0
        hsa_signal_t dummy_signal;
        hsa_amd_signal_create_on_agent(
            0, 0, nullptr, _air_host_active_segment.agent, 0, &dummy_signal);
        dummy_signal.handle =
            0; // The barrier and packet will ignore a signal with handle of 0
        signals_in_pkt.push_back(dummy_signal);
      }
    }
    if (non_zero) {

      // Submit a barrier packet for 5 signals that we are waiting on
      uint64_t wr_idx =
          hsa_queue_add_write_index_relaxed(_air_host_active_segment.q, 1);
      uint64_t packet_id = wr_idx % _air_host_active_segment.q->size;
      hsa_barrier_and_packet_t barrier_pkt;
      air_packet_barrier_and(&barrier_pkt, signals_in_pkt[0], signals_in_pkt[1],
                             signals_in_pkt[2], signals_in_pkt[3],
                             signals_in_pkt[4]);
      hsa_amd_signal_create_on_agent(1, 0, nullptr,
                                     _air_host_active_segment.agent, 0,
                                     &barrier_pkt.completion_signal);
      air_queue_dispatch(_air_host_active_segment.q, packet_id, wr_idx,
                         &barrier_pkt);

      // Put it in a vector of barrier packets so we can wait on all of them
      // after they are submitted
      packets.push_back(barrier_pkt);
    }

    // Remove the 5 signals from our vector and keep going if we have more
    signals.resize(signals.size() - 5);
  }

  // Submit each packet and delete the completion signal
  for (auto p : packets) {
    air_queue_wait(q, &p);
    hsa_signal_destroy(p.completion_signal);
  }

  return 0;
}

uint64_t air_get_tile_addr(uint32_t col, uint32_t row) {
  if (_air_host_active_libxaie == NULL)
    return -1;
  return _XAie_GetTileAddr(&(_air_host_active_libxaie->DevInst), row, col);
}

/// Read the AIE registers at the given physical address.
uint32_t air_read32(uint64_t addr) {
  if (_air_host_active_libxaie == NULL)
    return -1;
  uint32_t val;
  XAie_Read32(&(_air_host_active_libxaie->DevInst), addr, &val);
  return val;
}

/// Write the AIE registers at the given physical address.
/// It's almost always better to use some more indirect method of accessing
/// configuration registers, but this is provided as a last resort.
void air_write32(uint64_t addr, uint32_t val) {
  if (_air_host_active_libxaie == NULL)
    return;
  XAie_Write32(&(_air_host_active_libxaie->DevInst), addr, val);
}

extern "C" {

uint64_t _mlir_ciface___airrt_herd_load(const char *name) {
  return air_herd_load(name);
}

uint64_t _mlir_ciface___airrt_segment_load(const char *name) {
  return air_segment_load(name);
}

void _mlir_ciface___airrt_wait_all_0_0() { return; }
void _mlir_ciface___airrt_wait_all_0_1(uint64_t e0) {
  std::vector<uint64_t> events{e0, 0, 0, 0, 0};
  air_wait_all(events);
  return;
}
void _mlir_ciface___airrt_wait_all_0_2(uint64_t e0, uint64_t e1) {
  std::vector<uint64_t> events{e0, e1, 0, 0, 0};
  air_wait_all(events);
  return;
}
void _mlir_ciface___airrt_wait_all_0_3(uint64_t e0, uint64_t e1, uint64_t e2) {
  std::vector<uint64_t> events{e0, e1, e2, 0, 0};
  air_wait_all(events);
  return;
}

uint64_t _mlir_ciface___airrt_wait_all_1_0() {
  std::vector<uint64_t> events{};
  return air_wait_all(events);
}
uint64_t _mlir_ciface___airrt_wait_all_1_1(uint64_t e0) {
  std::vector<uint64_t> events{e0, 0, 0, 0, 0};
  return air_wait_all(events);
}
uint64_t _mlir_ciface___airrt_wait_all_1_2(uint64_t e0, uint64_t e1) {
  std::vector<uint64_t> events{e0, e1, 0, 0, 0};
  return air_wait_all(events);
}
uint64_t _mlir_ciface___airrt_wait_all_1_3(uint64_t e0, uint64_t e1,
                                           uint64_t e2) {
  std::vector<uint64_t> events{e0, e1, e2, 0, 0};
  return air_wait_all(events);
}

} // extern C
