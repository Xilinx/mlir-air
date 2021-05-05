#include "air_host.h"

#include <dlfcn.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

// temporary solution to stash some state
extern "C" {
air_herd_desc_t *_air_host_active_herd = nullptr;
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

air_herd_handle_t
air_herd_load_from_file(const char* filename)
{
  air_herd_handle_t handle;
  void* _handle = dlopen(filename, RTLD_NOW);
  if (!_handle) {
    printf("%s\n",dlerror());
    return 0;
  }

  handle = (air_herd_handle_t)_handle;
  _air_host_active_herd = air_herd_get_desc(handle);
  assert(_air_host_active_herd);

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  assert(fd != -1 && "Failed to open /dev/mem");

  _air_host_bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE,
                                        MAP_SHARED, fd,
                                        AIR_VCK190_SHMEM_BASE+0x4000);
  assert(_air_host_bram_ptr && "Failed to map scratch bram location");

  return handle;
}

int32_t
air_herd_unload(air_herd_handle_t handle)
{
  if (!handle)
    return -1;
  
  if (air_herd_get_desc(handle) == _air_host_active_herd)
    _air_host_active_herd = nullptr;

  return dlclose((void*)handle);
}

air_herd_desc_t *
air_herd_get_desc(air_herd_handle_t handle)
{
  if (!handle) return nullptr;
  return (air_herd_desc_t*)dlsym((void*)handle, "__air_herd_descriptor");
}

