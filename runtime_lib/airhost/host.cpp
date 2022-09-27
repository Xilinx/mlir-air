//===- host.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include "air_host.h"

#include <dlfcn.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

#include <string>

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
air_rt_partition_desc_t _air_host_active_partition = {nullptr, nullptr};
aie_libxaie_ctx_t *_air_host_active_libxaie1 = nullptr;
uint32_t *_air_host_bram_ptr = nullptr;
uint64_t _air_host_bram_paddr = 0;
air_module_handle_t _air_host_active_module = (air_module_handle_t)nullptr;

}

aie_libxaie_ctx_t *
air_init_libxaie1()
{
  if (_air_host_active_libxaie1)
    return _air_host_active_libxaie1;

  aie_libxaie_ctx_t *xaie =
    (aie_libxaie_ctx_t*)malloc(sizeof(aie_libxaie_ctx_t));
  if (!xaie)
    return 0;

  xaie->AieConfigPtr.AieGen = XAIE_DEV_GEN_AIE;
  xaie->AieConfigPtr.BaseAddr = XAIE_BASE_ADDR;
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

  _air_host_active_libxaie1 = xaie;
  return xaie;
}

void
air_deinit_libxaie1(aie_libxaie_ctx_t *xaie)
{
  if (xaie == _air_host_active_libxaie1) {
    XAie_Finish(&(xaie->DevInst));
    _air_host_active_libxaie1 = nullptr;
  }
  free(xaie);
}

air_module_handle_t
air_module_load_from_file(const char* filename, queue_t *q)
{

  if (_air_host_active_module)
    air_module_unload(_air_host_active_module);

  air_module_handle_t handle;
  void* _handle = dlopen(filename, RTLD_NOW);
  if (!_handle) {
    printf("%s\n",dlerror());
    return 0;
  }
  _air_host_active_module = (air_module_handle_t)_handle;
  _air_host_active_herd = {q, nullptr};
  _air_host_active_partition = {q, nullptr};

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  assert(fd != -1 && "Failed to open /dev/mem");

  _air_host_bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE,
                                        MAP_SHARED, fd,
                                        AIR_BBUFF_BASE);
  _air_host_bram_paddr = AIR_BBUFF_BASE;
  assert(_air_host_bram_ptr && "Failed to map scratch bram location");

  return (air_module_handle_t)_handle;
}

int32_t
air_module_unload(air_module_handle_t handle)
{
  if (!handle)
    return -1;

  if (auto module_desc = air_module_get_desc(handle)) {
    for (int i=0; i<module_desc->partition_length; i++) {
      for (int j=0; j<module_desc->partition_descs[i]->herd_length; j++) {
        auto herd_desc = module_desc->partition_descs[i]->herd_descs[j];
        if (herd_desc == _air_host_active_herd.herd_desc) {
          _air_host_active_herd = {nullptr, nullptr};
          _air_host_active_partition = {nullptr, nullptr};
        }
      }
    }
  }
  if (_air_host_active_module == handle) {
    _air_host_active_module = (air_module_handle_t)nullptr;
    munmap(_air_host_bram_ptr,0x8000);
    _air_host_bram_paddr = 0;
  }

  return dlclose((void*)handle);
}

air_herd_desc_t *
air_herd_get_desc(air_module_handle_t handle, air_partition_desc_t *partition_desc, const char *herd_name)
{
  if (!handle) return nullptr;
  if (!partition_desc) return nullptr;

  auto module_desc = air_module_get_desc(handle);
  if (!module_desc)
    return nullptr;

  if (!air_partition_get_desc(handle, partition_desc->name))
    return nullptr;

  for (int i=0; i<partition_desc->herd_length; i++) {
    auto herd_desc = partition_desc->herd_descs[i];
    if (!strncmp(herd_name, herd_desc->name, herd_desc->name_length))
      return herd_desc;
  }
  return nullptr;
}

air_partition_desc_t *
air_partition_get_desc(air_module_handle_t handle, const char *partition_name)
{
  if (!handle) return nullptr;

  auto module_desc = air_module_get_desc(handle);
  if (!module_desc) return nullptr;

  for (int i=0; i<module_desc->partition_length; i++) {
    auto partition_desc = module_desc->partition_descs[i];
    if (!strncmp(partition_name, partition_desc->name,
                 partition_desc->name_length)) {
      return partition_desc;
    }
  }
  return nullptr;
}

air_module_desc_t *
air_module_get_desc(air_module_handle_t handle)
{
  if (!handle) return nullptr;
  return (air_module_desc_t*)dlsym((void*)handle, "__air_module_descriptor");
}

uint64_t air_partition_load(const char *name) {

  assert(_air_host_active_libxaie1);

  XAie_Finish(&(_air_host_active_libxaie1->DevInst));
  XAie_CfgInitialize(&(_air_host_active_libxaie1->DevInst),
                     &(_air_host_active_libxaie1->AieConfigPtr));
  XAie_PmRequestTiles(&(_air_host_active_libxaie1->DevInst), NULL, 0);

  auto partition_desc = air_partition_get_desc(_air_host_active_module, name);
  if (!partition_desc) {
    printf("Failed to locate partition descriptor '%s'!\n", name);
    assert(0);
  }
  std::string partition_name(partition_desc->name, partition_desc->name_length);

  std::string func_name = "__airrt_" + partition_name + "_aie_functions";
  air_rt_aie_functions_t *mlir = (air_rt_aie_functions_t *)dlsym(
      (void *)_air_host_active_module, func_name.c_str());

  if (mlir) {
    // printf("configuring partition: '%s'\n", partition_name.c_str());
    assert(mlir->configure_cores);
    assert(mlir->configure_switchboxes);
    assert(mlir->initialize_locks);
    assert(mlir->configure_dmas);
    assert(mlir->start_cores);
    mlir->configure_cores(_air_host_active_libxaie1);
    mlir->configure_switchboxes(_air_host_active_libxaie1);
    mlir->initialize_locks(_air_host_active_libxaie1);
    mlir->configure_dmas(_air_host_active_libxaie1);
    mlir->start_cores(_air_host_active_libxaie1);
  } else {
    printf("Failed to locate partition '%s' configuration functions!\n",
           partition_name.c_str());
    assert(0);
  }
  _air_host_active_partition.partition_desc = partition_desc;
  return 0;
}

uint64_t
air_herd_load(const char *name) {

  // If no partition is loaded, load the partition associated with this herd
  if (!_air_host_active_partition.partition_desc) {
    bool loaded = false;
    if (auto module_desc = air_module_get_desc(_air_host_active_module)) {
      for (int i = 0; !loaded && i < module_desc->partition_length; i++) {
        for (int j = 0;
             !loaded && j < module_desc->partition_descs[i]->herd_length; j++) {
          auto herd_desc = module_desc->partition_descs[i]->herd_descs[j];
          // use the partition of the first herd with a matching name
          if (!strncmp(name, herd_desc->name, herd_desc->name_length)) {
            air_partition_load(module_desc->partition_descs[i]->name);
            loaded = true; // break
          }
        }
      }
    }
  }
  auto herd_desc = air_herd_get_desc(
      _air_host_active_module, _air_host_active_partition.partition_desc, name);
  // In some scenarios load_partition is not called. This is a temporary hack
  // to support that case.
  if (!herd_desc) {
    if (_air_host_active_partition.partition_desc) {
      _air_host_active_partition.partition_desc = 0;
      return air_herd_load(name);
    }
    printf("Failed to locate herd descriptor '%s'!\n",name);
    assert(0);
  }
  _air_host_active_herd.herd_desc = herd_desc;

  return 0;
}
