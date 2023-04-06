//===- host.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air_host.h"
#include "air_host_impl.h"
#include "test_library.h"

#ifdef AIR_PCIE
#include "utility.hpp"
#endif

#include <assert.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream> // ifstream
#include <iomanip> // setbase()
#include <iostream>
#include <stdio.h>
#include <string>
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

// temporary solution to stash some state
extern "C" {

air_rt_herd_desc_t _air_host_active_herd = {nullptr, nullptr};
air_rt_segment_desc_t _air_host_active_segment = {nullptr, nullptr};
aie_libxaie_ctx_t *_air_host_active_libxaie = nullptr;
uint32_t *_air_host_bram_ptr = nullptr;
uint64_t _air_host_bram_paddr = 0;
air_module_handle_t _air_host_active_module = (air_module_handle_t) nullptr;

const char vck5000_driver_name[] = "/dev/amdair";
}

#ifdef AIR_PCIE
volatile void *_mapped_aie_base = nullptr;
std::vector<air_physical_device_t> physical_devices;
#endif

hsa_status_t air_init() {
  printf("%s\n", __func__);
#ifdef AIR_PCIE
  hsa_status_t hsa_ret = air_get_physical_devices();

  if (hsa_ret != HSA_STATUS_SUCCESS) {
    std::cerr << "air_get_physical_devices failed" << std::endl;
    return hsa_ret;
  }
#endif

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
#ifdef AIR_PCIE

  if (device_id >= physical_devices.size()) {
    printf("[ERROR] No device id %d in system\n", device_id);
    return (air_libxaie_ctx_t) nullptr;
  }

  std::string aie_bar = air_get_aie_bar(device_id);

  int fda;
  if ((fda = open(vck5000_driver_name, O_RDWR | O_SYNC)) == -1) {
    printf("[ERROR] Failed to open device file\n");
    return (air_libxaie_ctx_t) nullptr;
  }

  // Map the memory region into userspace
  _mapped_aie_base = mmap(NULL,                   // virtual address
                          0x20000000,             // length
                          PROT_READ | PROT_WRITE, // prot
                          MAP_SHARED,             // flags
                          fda,                    // device fd
                          0x100000);              // offset
  if (_mapped_aie_base == MAP_FAILED) {
    printf("[ERROR] Failed mapping AIE BAR\n");
    return (air_libxaie_ctx_t) nullptr;
  }
  xaie->AieConfigPtr.BaseAddr = (uint64_t)_mapped_aie_base;
#else
  xaie->AieConfigPtr.BaseAddr = XAIE_BASE_ADDR;
#endif
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

  XAie_CfgInitialize(&(xaie->DevInst), &(xaie->AieConfigPtr));
  XAie_PmRequestTiles(&(xaie->DevInst), NULL, 0);

  _air_host_active_libxaie = xaie;
  return (air_libxaie_ctx_t)xaie;
}

void air_deinit_libxaie(air_libxaie_ctx_t _xaie) {
  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)_xaie;
  if (xaie == _air_host_active_libxaie) {
    XAie_Finish(&(xaie->DevInst));
#ifdef AIR_PCIE
    munmap(const_cast<void *>(_mapped_aie_base), 0x20000000);
    _mapped_aie_base = nullptr;
#endif
    _air_host_active_libxaie = nullptr;
  }
  free(xaie);
}

air_module_handle_t air_module_load_from_file(const char *filename, queue_t *q,
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
  _air_host_active_herd = {q, nullptr};
  _air_host_active_segment = {q, nullptr};

#ifdef AIR_PCIE

  if (device_id >= physical_devices.size()) {
    printf("[ERROR] No device id %d in system\n", device_id);
    return 0;
  }

  int fd = open(vck5000_driver_name, O_RDWR | O_SYNC);
  assert(fd != -1 && "Failed to open bram fd");

  _air_host_bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, 0x1C0000);
  _air_host_bram_paddr = AIR_BBUFF_BASE;
#else

#ifndef __aarch64__
  printf("[ERROR] Attempting to map /dev/mem on x86. Please define AIR_PCIE "
         "when compiling\n");
  return 0;
#endif

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  assert(fd != -1 && "Failed to open bram fd");

  _air_host_bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, AIR_BBUFF_BASE);
  _air_host_bram_paddr = AIR_BBUFF_BASE;
#endif
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
          _air_host_active_herd = {nullptr, nullptr};
          _air_host_active_segment = {nullptr, nullptr};
        }
      }
    }
  }
  if (_air_host_active_module == handle) {
    _air_host_active_module = (air_module_handle_t) nullptr;
    munmap(_air_host_bram_ptr, 0x8000);
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

#ifdef AIR_PCIE
  XAie_Finish(&(_air_host_active_libxaie->DevInst));
  XAie_CfgInitialize(&(_air_host_active_libxaie->DevInst),
                     &(_air_host_active_libxaie->AieConfigPtr));
  XAie_PmRequestTiles(&(_air_host_active_libxaie->DevInst), NULL, 0);

  uint64_t wr_idx = queue_add_write_index(_air_host_active_segment.q, 1);
  uint64_t packet_id = wr_idx % _air_host_active_segment.q->size;
  dispatch_packet_t *shim_pkt =
      (dispatch_packet_t *)(_air_host_active_segment.q->base_address_vaddr) +
      packet_id;
  air_packet_device_init(shim_pkt, XAIE_NUM_COLS);

  wr_idx = queue_add_write_index(_air_host_active_segment.q, 1);
  packet_id = wr_idx % _air_host_active_segment.q->size;
  dispatch_packet_t *herd_pkt =
      (dispatch_packet_t *)(_air_host_active_segment.q->base_address_vaddr) +
      packet_id;
  air_packet_herd_init(herd_pkt, 0, 0, 50, 1, 8);
  air_queue_dispatch_and_wait(_air_host_active_segment.q, wr_idx, herd_pkt);

#else
  XAie_Finish(&(_air_host_active_libxaie->DevInst));
  XAie_CfgInitialize(&(_air_host_active_libxaie->DevInst),
                     &(_air_host_active_libxaie->AieConfigPtr));
  XAie_PmRequestTiles(&(_air_host_active_libxaie->DevInst), NULL, 0);
#endif

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

#ifdef AIR_PCIE
hsa_status_t air_get_physical_devices() {

  struct stat st;

  // Skip device enumeration if it is already done
  if (physical_devices.size() != 0)
    return HSA_STATUS_SUCCESS;

  air_physical_device_t temp_physical_device;
  physical_devices.push_back(temp_physical_device);

  return HSA_STATUS_SUCCESS;
}
#endif

hsa_status_t air_iterate_agents(hsa_status_t (*callback)(air_agent_t agent,
                                                         void *data),
                                void *data) {

  uint64_t total_controllers = 0;

#ifdef AIR_PCIE
  air_agent_t a;
  a.handle = reinterpret_cast<uintptr_t>(0xBADF00DUL);
  callback(a, data);
  return HSA_STATUS_SUCCESS;
#else

#ifndef __aarch64__
  printf("[ERROR] Attempting to map /dev/mem on x86. Please define AIR_PCIE "
         "when compiling\n");
  return HSA_STATUS_ERROR;
#endif

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR;

  uint64_t *bram_base =
      reinterpret_cast<uint64_t *>(mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, AIR_VCK190_SHMEM_BASE));
  if (bram_base == MAP_FAILED) {
    printf("[ERROR] can't map BRAM\n");
    return HSA_STATUS_ERROR;
  }

  total_controllers = bram_base[65];
  if (total_controllers < 1) {
    std::cerr << "No agents found" << std::endl;
    return HSA_STATUS_ERROR;
  }

  uint64_t *base_addr = reinterpret_cast<uint64_t *>(AIR_VCK190_SHMEM_BASE);
  for (int i = 0; i < total_controllers; i++) {
    air_agent_t a;
    a.handle = reinterpret_cast<uintptr_t>(&base_addr[i]);
    callback(a, data);
  }

  auto res = munmap(bram_base, 0x1000);
  if (res) {
    std::cerr << "Could not munmap" << std::endl;
    return HSA_STATUS_ERROR;
  }
#endif

  return HSA_STATUS_SUCCESS;
}

#ifdef AIR_PCIE
const char *air_get_driver_name(void) { return vck5000_driver_name; }

std::string air_get_ddr_bar(uint32_t device_id) {
  if (device_id >= physical_devices.size()) {
    printf("[ERROR] Attempting to grab BAR of device %d which does not exist\n",
           device_id);
    return "";
  }
  return std::string(physical_devices.at(device_id).dram_bar_path);
}

std::string air_get_aie_bar(uint32_t device_id) {
  if (device_id >= physical_devices.size()) {
    printf("[ERROR] Attempting to grab BAR of device %d which does not exist\n",
           device_id);
    return "";
  }
  return std::string(physical_devices.at(device_id).aie_bar_path);
}
std::string air_get_bram_bar(uint32_t device_id) {
  if (device_id >= physical_devices.size()) {
    printf("[ERROR] Attempting to grab BAR of device %d which does not exist\n",
           device_id);
    return "";
  }
  return std::string(physical_devices.at(device_id).bram_bar_path);
}
#endif

uint64_t air_wait_all(std::vector<uint64_t> &signals) {
  queue_t *q = _air_host_active_segment.q;
  if (!q) {
    printf("WARNING: no queue provided, air_wait_all will return without "
           "waiting\n");
    return 0;
  }

  std::vector<dispatch_packet_t *> packets;
  while (signals.size()) {
    if (signals.size() < 5)
      signals.resize(5, 0);

    std::vector<uint64_t> addrs;
    bool non_zero = false;
    for (auto s : signals) {
      if (s) {
        addrs.push_back(((signal_t *)s)->handle);
        non_zero = true;
      } else {
        addrs.push_back(AIR_VCK190_SHMEM_BASE + MB_SHMEM_SIGNAL_OFFSET);
      }
    }
    if (non_zero) {
      uint64_t wr_idx = queue_add_write_index(q, 1);
      uint64_t packet_id = wr_idx % q->size;
      dispatch_packet_t *barrier_pkt =
          (dispatch_packet_t *)(q->base_address_vaddr) + packet_id;
      air_packet_barrier_and((barrier_and_packet_t *)barrier_pkt, addrs[0],
                             addrs[1], addrs[2], addrs[3], addrs[4]);
      signal_create(1, 0, NULL, (signal_t *)&barrier_pkt->completion_signal);
      air_queue_dispatch(q, wr_idx, barrier_pkt);
      packets.push_back(barrier_pkt);
    }
    signals.resize(signals.size() - 5);
  }

  for (auto p : packets)
    air_queue_wait(q, p);

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
