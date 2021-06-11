#include "air_host.h"

#include <dlfcn.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

// temporary solution to stash some state
extern "C" {
air_rt_herd_desc_t _air_host_active_herd = {nullptr, nullptr};
air_libxaie1_ctx_t *_air_host_active_libxaie1 = nullptr;
uint32_t *_air_host_bram_ptr = nullptr;
}

air_libxaie1_ctx_t *
air_init_libxaie1()
{
  air_libxaie1_ctx_t *xaie =
    (air_libxaie1_ctx_t*)malloc(sizeof(air_libxaie1_ctx_t));
  if (!xaie)
    return 0;

#ifdef AIR_LIBXAIE_ENABLE
  XAIEGBL_HWCFG_SET_CONFIG((&xaie->AieConfig),
                           XAIE_NUM_ROWS, XAIE_NUM_COLS, 0x800);
  XAieGbl_HwInit(&xaie->AieConfig);
  xaie->AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&xaie->AieInst,
                        &xaie->TileInst[0][0], xaie->AieConfigPtr);
#else
  printf("%s : FATAL ERROR : libxaie not enabled\n", __func__);
  assert(0 && "AIR_LIBXAIE_ENABLE not defined");
#endif

  _air_host_active_libxaie1 = xaie;
  return xaie;
}

void
air_deinit_libxaie1(air_libxaie1_ctx_t *xaie)
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

  handle = (air_module_handle_t)_handle;
  auto module_desc = air_module_get_desc(handle);

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

  return handle;
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

  return dlclose((void*)handle);
}

air_herd_desc_t *
air_herd_get_desc(air_module_handle_t handle, const char *herd_name)
{
  if (!handle) return nullptr;
  auto module_desc = air_module_get_desc(handle);
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
