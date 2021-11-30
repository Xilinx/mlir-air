#include "air_host.h"

#include <dlfcn.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

#include <string>

#define XAIE_NUM_ROWS 8
#define XAIE_NUM_COLS 50

// temporary solution to stash some state
extern "C" {

air_rt_herd_desc_t _air_host_active_herd = {nullptr, nullptr};
aie_libxaie_ctx_t *_air_host_active_libxaie1 = nullptr;
uint32_t *_air_host_bram_ptr = nullptr;
air_module_handle_t _air_host_active_module = (air_module_handle_t)nullptr;

}

aie_libxaie_ctx_t *
air_init_libxaie1()
{
  aie_libxaie_ctx_t *xaie =
    (aie_libxaie_ctx_t*)malloc(sizeof(aie_libxaie_ctx_t));
  if (!xaie)
    return 0;

  XAIEGBL_HWCFG_SET_CONFIG((&xaie->AieConfig),
                           XAIE_NUM_ROWS, XAIE_NUM_COLS, 0x800);
  XAieGbl_HwInit(&xaie->AieConfig);
  xaie->AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&xaie->AieInst,
                        &xaie->TileInst[0][0], xaie->AieConfigPtr);

  _air_host_active_libxaie1 = xaie;
  return xaie;
}

void
air_deinit_libxaie1(aie_libxaie_ctx_t *xaie)
{
  if (xaie == _air_host_active_libxaie1)
    _air_host_active_libxaie1 = nullptr;
  free(xaie);
}

air_module_handle_t
air_module_load_from_file(const char* filename, queue_t *q)
{
  air_module_handle_t handle;
  void* _handle = dlopen(filename, RTLD_NOW);
  if (!_handle) {
    printf("%s\n",dlerror());
    return 0;
  }

  if (_air_host_active_module)
    air_module_unload(_air_host_active_module);
  _air_host_active_module = (air_module_handle_t)_handle;

  auto module_desc = air_module_get_desc(_air_host_active_module);

  if (module_desc->length)
    _air_host_active_herd.herd_desc = module_desc->herd_descs[0];
  _air_host_active_herd.q = q;

  assert(_air_host_active_herd.herd_desc);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  assert(fd != -1 && "Failed to open /dev/mem");

  _air_host_bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE,
                                        MAP_SHARED, fd,
                                        AIR_VCK190_SHMEM_BASE+0x4000);
  assert(_air_host_bram_ptr && "Failed to map scratch bram location");

  return (air_module_handle_t)_handle;
}

int32_t
air_module_unload(air_module_handle_t handle)
{
  if (!handle)
    return -1;

  auto module_desc = air_module_get_desc(handle);
  for (int i=0; i<module_desc->length; i++) {
    auto herd_desc = module_desc->herd_descs[i];
    if (herd_desc == _air_host_active_herd.herd_desc) {
      _air_host_active_herd.herd_desc = nullptr;
      _air_host_active_herd.q = nullptr;
    }
  }
  if (_air_host_active_module == handle)
    _air_host_active_module = (air_module_handle_t)nullptr;

  return dlclose((void*)handle);
}

air_herd_desc_t *
air_herd_get_desc(air_module_handle_t handle, const char *herd_name)
{
  if (!handle) return nullptr;

  auto module_desc = air_module_get_desc(handle);
  if (!module_desc) return nullptr;

  for (int i=0; i<module_desc->length; i++) {
    auto herd_desc = module_desc->herd_descs[i];
    if (!strncmp(herd_name, herd_desc->name, herd_desc->name_length))
      return herd_desc;
  }
  return nullptr;
}

air_module_desc_t *
air_module_get_desc(air_module_handle_t handle)
{
  if (!handle) return nullptr;
  return (air_module_desc_t*)dlsym((void*)handle, "__air_module_descriptor");
}

int32_t
air_herd_load(const char *name) {
  auto herd_desc = air_herd_get_desc(_air_host_active_module, name);
  if (!herd_desc) return -1;
  
  // bool configured = (_air_host_active_herd.herd_desc == herd_desc);
  // if (configured) return 0;
  
  _air_host_active_herd.herd_desc = herd_desc;

  std::string func_name = "__airrt_" +
                          std::string(name) +
                          "_aie_functions";
  air_rt_aie_functions_t *mlir = 
    (air_rt_aie_functions_t *)dlsym((void*)_air_host_active_module,
                                    func_name.c_str());

  if (mlir) {
    printf("configuring herd: '%s'\n",name);
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
  }

  return 0;
}